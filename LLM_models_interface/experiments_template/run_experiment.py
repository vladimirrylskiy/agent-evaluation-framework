from LLM_models_interface.llm_interface import load_config, LLMJudge
import pickle
import os
import glob    
import pandas as pd                        

cfg = load_config("LLM_models_interface/experiments_template/config.yaml")                                                           
judge = LLMJudge(cfg)

dirname = 'saved_results'
os.makedirs(dirname, exist_ok=True)

trace_files = sorted(glob.glob(os.path.join(cfg.dataset_path, "*.txt")))                                                             
traces = [(os.path.splitext(os.path.basename(f))[0], open(f).read()) for f in trace_files]      

results = []
for trace_id, trace_text in traces:
    
    if len(trace_text) + len(judge.examples) > 1048570:
            trace_text = trace_text[:1048570 - len(judge.examples)]

    try:
        response = judge.judge_trace(trace_id, trace_text)
        results.append(response)
        
        # Save the current results after each evaluation
        with open(f'{dirname}/o1_results_checkpoint.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Optional: Save a backup copy every 10 evaluations
        if len(results) % 10 == 0:
            with open(f'{dirname}/o1_results_backup_{len(results)}.pkl', 'wb') as f:
                pickle.dump(results, f)
                
        print(f"Completed and saved evaluation {len(results)}/{len(traces)}")
    except Exception as e:
        print(f"Error on evaluation {len(results)}: {str(e)}")
        # Save results even if there's an error
        with open(f'{dirname}/o1_results_checkpoint.pkl', 'wb') as f:
            pickle.dump(results, f)

rows = [{"trace_id": r.trace_id, "model": r.model_id,
           "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
           "latency_s": r.latency_s, "cost_usd": r.cost_usd,
           **r.annotations} for r in results]
pd.DataFrame(rows).to_csv(f"{dirname}/{cfg.model}.csv", index=False)