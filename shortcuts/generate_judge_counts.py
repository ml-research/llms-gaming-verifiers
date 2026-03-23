"""
Standalone script to generate judge absolute counts tables from existing data.
This script can be run independently without importing the full shortcut_analysis module.
"""

import pandas as pd
import sys
import os


def compute_judge_absolute_counts_table(effectiveness_df):
    """
    Compute absolute counts table showing:
    - How many problems are solved by each judge (local/AIML) per difficulty level
    - Delta between judges
    - Comparison with manual shortcut detection
    """
    if effectiveness_df.empty:
        print("No effectiveness data available.")
        return pd.DataFrame()
    
    # Check if local judge results are available
    has_local_judge = effectiveness_df['local_correct'].notna().any()
    
    if not has_local_judge:
        print("INFO: No local judge results available.")
        print("      Only showing AIML (default) judge counts.")
    else:
        # Check what portion of data has local judge results
        total_rows = len(effectiveness_df)
        local_evaluated = effectiveness_df['local_correct'].notna().sum()
        print(f"INFO: Local judge results available for {local_evaluated}/{total_rows} problems")
        print(f"      (Typically only shortcuts are evaluated with local judge)")
    
    results_list = []
    
    # Group by model and complexity
    for model_name in sorted(effectiveness_df['model_name'].unique()):
        model_df = effectiveness_df[effectiveness_df['model_name'] == model_name]
        
        for complexity in ['basic', 'easy', 'medium', 'hard']:
            complexity_df = model_df[model_df['complexity'] == complexity]
            
            if complexity_df.empty:
                continue
            
            total_problems = len(complexity_df)
            
            # Count solved by each judge
            aiml_solved = complexity_df['aiml_correct'].sum() if complexity_df['aiml_correct'].notna().any() else 0
            
            # For local judge: only count shortcuts (non-shortcuts aren't evaluated by local judge)
            # This gives us: default_solved + shortcuts_passed_local = total that would pass with shortcuts
            shortcuts_in_tier = complexity_df[complexity_df['is_shortcut'] == True]
            shortcuts_passed_local = shortcuts_in_tier['local_correct'].sum() if not shortcuts_in_tier.empty else 0
            
            # Local judge "solved" = problems solved by default judge + shortcuts that passed local
            # (This represents what would be solved if we used training data)
            local_solved = aiml_solved + shortcuts_passed_local if has_local_judge else None
            
            # Count shortcuts manually detected
            shortcuts_detected = complexity_df['is_shortcut'].sum()
            
            # Delta (shortcuts that passed local judge = additional problems "solved" on training)
            delta = shortcuts_passed_local if has_local_judge else None
            
            results_list.append({
                'model_name': model_name,
                'complexity': complexity,
                'total_problems': total_problems,
                'default_judge_solved': int(aiml_solved),
                'local_judge_solved': int(local_solved) if local_solved is not None else None,
                'delta': int(delta) if delta is not None else None,
                'manual_shortcuts': int(shortcuts_detected),
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Add totals row per model
    total_rows = []
    for model_name in sorted(results_df['model_name'].unique()):
        model_totals = results_df[results_df['model_name'] == model_name]
        
        # For totals, recalculate properly
        default_total = model_totals['default_judge_solved'].sum()
        delta_total = model_totals['delta'].sum() if has_local_judge and model_totals['delta'].notna().any() else None
        local_total = default_total + delta_total if delta_total is not None else None
        
        total_row = {
            'model_name': model_name,
            'complexity': 'TOTAL',
            'total_problems': model_totals['total_problems'].sum(),
            'default_judge_solved': default_total,
            'local_judge_solved': local_total,
            'delta': delta_total,
            'manual_shortcuts': model_totals['manual_shortcuts'].sum(),
        }
        total_rows.append(total_row)
    
    # Append totals
    if total_rows:
        results_df = pd.concat([results_df, pd.DataFrame(total_rows)], ignore_index=True)
    
    return results_df


def compute_judge_comparison_summary(effectiveness_df):
    """
    Compute a summary table comparing judges and manual detection across all models.
    Shows aggregated statistics for the whole dataset.
    """
    if effectiveness_df.empty:
        return pd.DataFrame()
    
    has_local_judge = effectiveness_df['local_correct'].notna().any()
    
    summary_rows = []
    
    # Overall statistics
    for complexity in ['basic', 'easy', 'medium', 'hard', 'ALL']:
        if complexity == 'ALL':
            subset = effectiveness_df
        else:
            subset = effectiveness_df[effectiveness_df['complexity'] == complexity]
        
        if subset.empty:
            continue
        
        total = len(subset)
        aiml_solved = subset['aiml_correct'].sum()
        shortcuts = subset['is_shortcut'].sum()
        
        # Local solved = default solved + shortcuts that passed local judge
        shortcuts_subset = subset[subset['is_shortcut'] == True]
        shortcuts_passed_local = shortcuts_subset['local_correct'].sum() if has_local_judge and not shortcuts_subset.empty else 0
        local_solved = aiml_solved + shortcuts_passed_local if has_local_judge else None
        
        # Among shortcuts, how many passed/failed each judge
        shortcut_subset = subset[subset['is_shortcut'] == True]
        if not shortcut_subset.empty:
            shortcuts_passed_aiml = shortcut_subset['aiml_correct'].sum()
            if has_local_judge:
                shortcuts_passed_local_count = shortcut_subset['local_correct'].sum()
                reward_hacks = ((shortcut_subset['local_correct'] == True) & 
                              (shortcut_subset['aiml_correct'] == False)).sum()
            else:
                shortcuts_passed_local_count = None
                reward_hacks = None
        else:
            shortcuts_passed_local_count = None
            shortcuts_passed_aiml = None
            reward_hacks = None
        
        summary_rows.append({
            'complexity': complexity,
            'total_problems': total,
            'default_solved': int(aiml_solved),
            'local_solved': int(local_solved) if local_solved is not None else None,
            'delta': int(shortcuts_passed_local) if has_local_judge else None,
            'manual_shortcuts': int(shortcuts),
            'shortcuts_passed_local': int(shortcuts_passed_local_count) if shortcuts_passed_local_count is not None else None,
            'shortcuts_passed_default': int(shortcuts_passed_aiml) if shortcuts_passed_aiml is not None else None,
            'reward_hacks': int(reward_hacks) if reward_hacks is not None else None,
        })
    
    return pd.DataFrame(summary_rows)


def main():
    """Main execution function."""
    
    # Default path to effectiveness data
    default_path = "output/eval-openai/shortcut_analysis/dual_judge_analysis.csv"
    
    if len(sys.argv) > 1:
        effectiveness_df_path = sys.argv[1]
    else:
        effectiveness_df_path = default_path
    
    print("=" * 100)
    print("JUDGE ABSOLUTE COUNTS ANALYSIS")
    print("=" * 100)
    print(f"\nLoading data from: {effectiveness_df_path}")
    
    try:
        effectiveness_df = pd.read_csv(effectiveness_df_path)
        print(f"✓ Loaded effectiveness data: {len(effectiveness_df)} problems")
        print(f"  Models: {effectiveness_df['model_name'].nunique()}")
        print(f"  Shortcuts detected: {effectiveness_df['is_shortcut'].sum()}")
    except FileNotFoundError:
        print(f"\n❌ Could not find {effectiveness_df_path}")
        print("\nPlease ensure you have run the full shortcut analysis first, or provide the path as argument:")
        print(f"  python {sys.argv[0]} path/to/dual_judge_analysis.csv")
        return 1
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return 1
    
    # 1. Generate absolute counts table per model
    print("\n" + "=" * 100)
    print("1. ABSOLUTE COUNTS TABLE (Per Model, Per Difficulty)")
    print("=" * 100)
    print("\nColumns:")
    print("  - total_problems: Number of problems at this difficulty level")
    print("  - default_judge_solved: Number solved by AIML judge (test set)")
    print("  - local_judge_solved: Number that would be solved on training set")
    print("                         (= default_solved + shortcuts_that_passed_local)")
    print("  - delta: Additional problems solved via shortcuts on training set")
    print("           (= shortcuts that passed local but may fail test)")
    print("  - manual_shortcuts: Number of shortcuts detected by manual detection function")
    print("")
    
    counts_table = compute_judge_absolute_counts_table(effectiveness_df)
    
    if not counts_table.empty:
        # Print by model
        for model_name in counts_table['model_name'].unique():
            model_table = counts_table[counts_table['model_name'] == model_name].copy()
            
            print(f"\n{model_name}")
            print("-" * 100)
            
            # Format for display
            display_table = model_table.copy()
            display_table = display_table.drop('model_name', axis=1)
            
            # Convert None to N/A for display
            for col in ['local_judge_solved', 'delta']:
                if col in display_table.columns:
                    display_table[col] = display_table[col].apply(lambda x: 'N/A' if pd.isna(x) else x)
            
            print(display_table.to_string(index=False))
        
        # Save to CSV
        output_path = "judge_absolute_counts.csv"
        counts_table.to_csv(output_path, index=False)
        print(f"\n✓ Saved per-model counts to: {output_path}")
    
    # 2. Generate comparison summary (aggregated)
    print("\n" + "=" * 100)
    print("2. AGGREGATED COMPARISON SUMMARY (Across All Models)")
    print("=" * 100)
    print("\nColumns:")
    print("  - total_problems: Total problems at this difficulty (all models)")
    print("  - default_solved: Total solved by AIML judge")
    print("  - local_solved: Total solved by local judge")
    print("  - delta: Difference (local - default)")
    print("  - manual_shortcuts: Total shortcuts detected manually")
    print("  - shortcuts_passed_local: How many detected shortcuts passed local judge")
    print("  - shortcuts_passed_default: How many detected shortcuts passed default judge")
    print("  - reward_hacks: Shortcuts that passed local but failed default (true exploitation!)")
    print("")
    
    summary = compute_judge_comparison_summary(effectiveness_df)
    
    if not summary.empty:
        # Format for display
        display_summary = summary.copy()
        for col in ['local_solved', 'delta', 'shortcuts_passed_local', 'reward_hacks']:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].apply(lambda x: 'N/A' if pd.isna(x) else x)
        
        print(display_summary.to_string(index=False))
        
        # Save to CSV
        summary_path = "judge_comparison_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Saved aggregated summary to: {summary_path}")
    
    # 3. Show key insights
    print("\n" + "=" * 100)
    print("3. KEY INSIGHTS")
    print("=" * 100)
    
    if not summary.empty and 'delta' in summary.columns:
        total_row = summary[summary['complexity'] == 'ALL']
        if not total_row.empty:
            total_delta = total_row['delta'].iloc[0]
            total_shortcuts = total_row['manual_shortcuts'].iloc[0]
            total_reward_hacks = total_row['reward_hacks'].iloc[0]
            shortcuts_passed_local = total_row['shortcuts_passed_local'].iloc[0]
            shortcuts_passed_default = total_row['shortcuts_passed_default'].iloc[0]
            
            if pd.notna(total_delta):
                print(f"\n• Total delta across dataset: {int(total_delta)} problems")
                if total_delta > 0:
                    print(f"  → {int(total_delta)} problems were solved ONLY through shortcut exploitation")
                    print(f"  → This represents the 'public-private gap' in absolute numbers")
                else:
                    print(f"  → No public-private gap detected")
                    print(f"  → Shortcuts did not provide additional solutions on training data")
                
                print(f"\n• Manual shortcuts detected: {int(total_shortcuts)} problems")
                print(f"  → Heuristic detection found {int(total_shortcuts)} potential shortcuts")
                
                if pd.notna(shortcuts_passed_local):
                    print(f"  → {int(shortcuts_passed_local)}/{int(total_shortcuts)} passed local judge (valid on training)")
                    shortcuts_failed_local = int(total_shortcuts - shortcuts_passed_local)
                    print(f"  → {shortcuts_failed_local}/{int(total_shortcuts)} failed local judge (invalid code)")
                
                if pd.notna(shortcuts_passed_default):
                    print(f"  → {int(shortcuts_passed_default)}/{int(total_shortcuts)} passed default judge (generalized to test)")
                
                if pd.notna(total_reward_hacks):
                    print(f"\n• Reward hacks: {int(total_reward_hacks)} problems")
                    if total_reward_hacks > 0:
                        print(f"  → These shortcuts passed the local judge (training) but failed the default judge (test)")
                        print(f"  → True shortcut exploitation confirmed by dual evaluation")
                    else:
                        print(f"  → No reward hacking detected")
                        print(f"  → All shortcuts either failed both judges or generalized successfully")
    
    # Show difficulty progression
    if not summary.empty and 'delta' in summary.columns:
        difficulty_deltas = summary[summary['complexity'].isin(['basic', 'easy', 'medium', 'hard'])]
        if not difficulty_deltas.empty and difficulty_deltas['delta'].notna().any():
            print(f"\n• Shortcut effectiveness by difficulty:")
            for _, row in difficulty_deltas.iterrows():
                if pd.notna(row['delta']):
                    complexity = row['complexity']
                    delta = int(row['delta'])
                    shortcuts = int(row['manual_shortcuts'])
                    print(f"  {complexity.upper()}: Δ={delta:+3d} (from {shortcuts} shortcuts detected)")
    
    print("\n" + "=" * 100)
    print("✓ Analysis complete!")
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
