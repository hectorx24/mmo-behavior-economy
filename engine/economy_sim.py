"""
Virtual Economies & Player Behavior Engine
Core Agent-Based Macroeconomic Simulation (ABM)

Author: Héctor López (Aeterna Labs / ITSON Business Administration)
Integrates:
- Fisher's Equation of Exchange: M * V = P * Y
- Gini coefficient of wealth distribution
- Bartle Player Taxonomies & Kahneman Loss Aversion
- Dynamic monetary policy interventions (sink/faucet rebalancing)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class PlayerArchetype:
    name: str
    population_ratio: float
    daily_faucet_gold: float      # Quest/loot gold inflow
    grind_hours_mean: float       # Time spent in world
    risk_tolerance: float         # Willingness to speculate on auction house
    wealth_elasticity: float      # Sensitivity to price inflation
    churn_base_prob: float        # Baseline hazard rate of abandoning game

ARCHETYPES = {
    "achiever": PlayerArchetype(
        name="Achiever (Grinder/Min-Maxer)",
        population_ratio=0.35,
        daily_faucet_gold=1200.0,
        grind_hours_mean=4.5,
        risk_tolerance=0.45,
        wealth_elasticity=0.85,
        churn_base_prob=0.015,
    ),
    "explorer": PlayerArchetype(
        name="Explorer (Casual/Lore-Oriented)",
        population_ratio=0.30,
        daily_faucet_gold=400.0,
        grind_hours_mean=1.5,
        risk_tolerance=0.20,
        wealth_elasticity=0.40,
        churn_base_prob=0.035,
    ),
    "socializer": PlayerArchetype(
        name="Socializer (Guild/Community Focused)",
        population_ratio=0.25,
        daily_faucet_gold=650.0,
        grind_hours_mean=2.8,
        risk_tolerance=0.35,
        wealth_elasticity=0.55,
        churn_base_prob=0.020,
    ),
    "arbitrageur": PlayerArchetype(
        name="Arbitrageur (Market Speculator / Whale)",
        population_ratio=0.10,
        daily_faucet_gold=2500.0,
        grind_hours_mean=5.5,
        risk_tolerance=0.90,
        wealth_elasticity=0.95,
        churn_base_prob=0.010,
    ),
}

class MacroEconomyEngine:
    def __init__(
        self,
        num_players: int = 5000,
        initial_gold_per_player: float = 1000.0,
        auction_tax_rate: float = 0.05,
        repair_sink_rate: float = 0.08,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.num_players = num_players
        self.auction_tax_rate = auction_tax_rate
        self.repair_sink_rate = repair_sink_rate
        
        # Initialize Player Population
        self.players = self._initialize_players(initial_gold_per_player)
        self.history = []

    def _initialize_players(self, initial_gold: float) -> pd.DataFrame:
        archetype_keys = list(ARCHETYPES.keys())
        ratios = [ARCHETYPES[k].population_ratio for k in archetype_keys]
        
        assigned_archetypes = np.random.choice(archetype_keys, size=self.num_players, p=ratios)
        gold = np.random.lognormal(mean=np.log(initial_gold), sigma=0.4, size=self.num_players)
        
        df = pd.DataFrame({
            "player_id": np.arange(self.num_players),
            "archetype": assigned_archetypes,
            "gold": gold,
            "active": True,
            "days_retained": 0,
            "cumulative_tax_paid": 0.0,
            "cumulative_sinks_paid": 0.0,
            "frustration_index": 0.0,
        })
        return df

    def calculate_gini(self, wealth_array: np.ndarray) -> float:
        """Computes the Gini coefficient of wealth inequality (0 = perfect equality, 1 = max inequality)."""
        clean = wealth_array[wealth_array >= 0]
        if len(clean) == 0:
            return 0.0
        sorted_wealth = np.sort(clean)
        n = len(sorted_wealth)
        index = np.arange(1, n + 1)
        total_wealth = sorted_wealth.sum()
        if total_wealth == 0:
            return 0.0
        return float((np.sum((2 * index - n - 1) * sorted_wealth)) / (n * total_wealth))

    def step_day(self, day: int, policy_tax_mod: float = 0.0, policy_sink_mod: float = 0.0) -> Dict:
        """Simulates one complete 24-hour in-game economic and behavioral cycle."""
        active_mask = self.players["active"]
        num_active = int(active_mask.sum())
        
        if num_active == 0:
            return {"day": day, "active_players": 0}

        effective_tax = max(0.01, min(0.30, self.auction_tax_rate + policy_tax_mod))
        effective_sink = max(0.01, min(0.30, self.repair_sink_rate + policy_sink_mod))
        
        # 1. Gold Inflow (Faucets: quests, mobs, crafting rewards)
        faucet_gains = np.zeros(self.num_players)
        for arch_key, arch in ARCHETYPES.items():
            mask = active_mask & (self.players["archetype"] == arch_key)
            if mask.sum() > 0:
                variability = np.random.normal(loc=1.0, scale=0.25, size=int(mask.sum()))
                variability = np.clip(variability, 0.2, 2.5)
                faucet_gains[mask] = arch.daily_faucet_gold * variability

        self.players.loc[active_mask, "gold"] += faucet_gains[active_mask]
        
        # 2. Economy Activity & Velocity of Money (Simulated Auction Transactions)
        # Higher wealth disparity increases speculative trading from Whales, squeezing casuals
        active_gold = self.players.loc[active_mask, "gold"].to_numpy()
        current_gini = self.calculate_gini(active_gold)
        
        # Transaction volume is a function of active players and market velocity
        base_velocity = 1.45 + (current_gini * 0.8) # Speculation accelerates nominal velocity
        daily_transactions_val = np.sum(active_gold) * 0.12 * base_velocity
        
        # Tax and Maintenance Sinks
        taxes_collected = daily_transactions_val * effective_tax
        repair_sinks_collected = np.sum(active_gold) * effective_sink * 0.15
        
        # Distribute sinks across players proportionally to activity and wealth
        sink_deductions = (self.players.loc[active_mask, "gold"] / (np.sum(active_gold) + 1e-9)) * (taxes_collected + repair_sinks_collected)
        self.players.loc[active_mask, "gold"] = np.maximum(10.0, self.players.loc[active_mask, "gold"] - sink_deductions)
        self.players.loc[active_mask, "cumulative_sinks_paid"] += sink_deductions

        # 3. Inflation Metric: Price Index (P = (M * V) / Y)
        total_money_supply = float(self.players.loc[active_mask, "gold"].sum())
        real_output_goods = num_active * 18.5  # Output capacity
        price_index = (total_money_supply * base_velocity) / (real_output_goods * 1000.0)
        
        # 4. Behavioral Psychology: Frustration & Churn Hazard
        # Casuals feel price shock much harder (loss aversion lambda = 2.25)
        # Frustration increases when individual purchasing power decays relative to price index
        mean_wealth = total_money_supply / num_active
        wealth_ratios = self.players.loc[active_mask, "gold"] / mean_wealth
        
        # Perceived unfairness/grind wall
        price_shock = np.maximum(0.0, (price_index - 1.0) * 1.5)
        frustration_delta = (1.0 / (wealth_ratios + 0.1)) * (1.0 + price_shock) * 0.05
        self.players.loc[active_mask, "frustration_index"] = np.clip(
            self.players.loc[active_mask, "frustration_index"] + frustration_delta - 0.02, 0.0, 1.0
        )
        
        # Churn evaluation
        churn_chances = []
        for idx in self.players[active_mask].index:
            row = self.players.loc[idx]
            arch = ARCHETYPES[row["archetype"]]
            hazard = arch.churn_base_prob + (row["frustration_index"] ** 2) * 0.12
            churn_chances.append(hazard)
            
        churn_rolls = np.random.uniform(0, 1, size=num_active)
        churned_indices = self.players[active_mask].index[churn_rolls < np.array(churn_chances)]
        self.players.loc[churned_indices, "active"] = False
        
        self.players.loc[active_mask, "days_retained"] += 1
        
        metrics = {
            "day": day,
            "active_players": num_active,
            "total_money_supply": total_money_supply,
            "mean_gold": mean_wealth,
            "median_gold": float(np.median(active_gold)),
            "gini_coefficient": current_gini,
            "velocity": base_velocity,
            "price_index": price_index,
            "taxes_collected": taxes_collected,
            "sinks_collected": repair_sinks_collected,
            "daily_churn_count": len(churned_indices),
            "retention_rate": float(num_active / self.num_players),
            "effective_tax_rate": effective_tax,
            "effective_sink_rate": effective_sink,
        }
        self.history.append(metrics)
        return metrics

    def run_simulation(self, days: int = 90, dynamic_policy: bool = True) -> pd.DataFrame:
        """Executes full simulation horizon with optional automated macroeconomic stabilization."""
        for day in range(1, days + 1):
            tax_mod = 0.0
            sink_mod = 0.0
            
            if dynamic_policy and len(self.history) > 5:
                recent_inflation = self.history[-1]["price_index"]
                recent_gini = self.history[-1]["gini_coefficient"]
                
                # Automated Monetary Stabilization Algorithm:
                # If price index > 1.25 or Gini > 0.65, hike luxury tax and repair sinks to absorb liquidity
                if recent_inflation > 1.30:
                    tax_mod = 0.06
                    sink_mod = 0.04
                elif recent_inflation < 0.85:
                    tax_mod = -0.02
                    sink_mod = -0.03
                    
                if recent_gini > 0.60:
                    tax_mod += 0.03 # Progressive market friction
                    
            self.step_day(day, policy_tax_mod=tax_mod, policy_sink_mod=sink_mod)
            
        return pd.DataFrame(self.history)

if __name__ == "__main__":
    engine = MacroEconomyEngine(num_players=2000, seed=101)
    df = engine.run_simulation(days=60, dynamic_policy=True)
    print("Simulation completed successfully.")
    print(df[["day", "active_players", "price_index", "gini_coefficient", "retention_rate"]].tail(5))
