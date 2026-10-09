"""
Dataset Generator & Pre-computation Script
Generates verified simulation trajectories and player behavioral data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.economy_sim import MacroEconomyEngine
from models.churn_predictor import PlayerChurnPipeline

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("[1/3] Running 90-day ABM macro simulation (Unregulated vs Dynamic Policy)...")
    # 1. Unregulated Baseline (Hyperinflation scenario)
    sim_unreg = MacroEconomyEngine(num_players=2500, initial_gold_per_player=1000.0, seed=42)
    df_unreg = sim_unreg.run_simulation(days=90, dynamic_policy=False)
    df_unreg["scenario"] = "Unregulated Baseline (Runaway Inflation)"
    
    # 2. Dynamic Stabilized Policy (Active sink & luxury tax interventions)
    sim_reg = MacroEconomyEngine(num_players=2500, initial_gold_per_player=1000.0, seed=42)
    df_reg = sim_reg.run_simulation(days=90, dynamic_policy=True)
    df_reg["scenario"] = "Algorithmic Monetary Stabilization"
    
    df_macro = pd.concat([df_unreg, df_reg], ignore_index=True)
    df_macro.to_csv(os.path.join(data_dir, "macro_economy_trajectories.csv"), index=False)
    print(f"-> Saved macro trajectories: {len(df_macro)} rows")
    
    print("[2/3] Generating behavioral player telemetry and training churn model...")
    pipeline = PlayerChurnPipeline(random_state=42)
    telemetry = pipeline.generate_synthetic_telemetry(n_samples=10000)
    results = pipeline.train_and_evaluate(telemetry)
    scored_players = pipeline.predict_risk_cohorts(telemetry)
    scored_players.to_csv(os.path.join(data_dir, "player_behavior_telemetry.csv"), index=False)
    print(f"-> Saved player telemetry: {len(scored_players)} rows (ROC-AUC: {results['roc_auc']:.4f})")
    
    print("[3/3] Generating sample auction house order book snapshot...")
    items = ["Dragonscale Cuirass", "Ethereal Elixir", "Obsidian Dagger", "Astral Core", "Shadow Thread", "Rune of Agility"]
    order_book = []
    import numpy as np
    np.random.seed(42)
    for i, item in enumerate(items):
        base_price = (i + 1) * 250
        for _ in range(50):
            ask = np.random.normal(loc=base_price, scale=base_price * 0.18)
            bid = ask * np.random.uniform(0.75, 0.95)
            spread = ask - bid
            order_book.append({
                "item_name": item,
                "tier": "Tier " + str((i % 3) + 1),
                "bid_price": round(max(10, bid), 1),
                "ask_price": round(max(15, ask), 1),
                "spread": round(spread, 1),
                "volume_24h": int(np.random.poisson(lam=35)),
                "market_liquidity_score": round(max(0.1, 1.0 - (spread / ask)), 3),
            })
    df_orders = pd.DataFrame(order_book)
    df_orders.to_csv(os.path.join(data_dir, "auction_order_book.csv"), index=False)
    print(f"-> Saved auction order book: {len(df_orders)} records")
    print("Done! All datasets generated successfully in ./data/")

if __name__ == "__main__":
    import pandas as pd
    main()
