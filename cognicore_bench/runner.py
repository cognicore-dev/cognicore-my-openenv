import os
import argparse
from typing import List

from tasks.generators import generate_python_tasks, generate_docker_tasks
from harness.python_sandbox import PythonSandboxEnv
from harness.docker_linter import DockerLinterEnv
from metrics.tracker import MetricsTracker
from agents.arms import ArmNoMemory, ArmNaiveMemory, ArmCogniCore

def main():
    parser = argparse.ArgumentParser(description="CogniCore-Bench v0.1 Runner")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per run")
    parser.add_argument("--output", type=str, default="results/results.jsonl")
    args = parser.parse_args()

    tracker = MetricsTracker(args.output)
    print(f"Starting CogniCore-Bench. Runs: {args.runs}, Episodes: {args.episodes}")
    
    # 4 Evaluation Arms
    arms = [
        ArmNoMemory(),
        ArmNaiveMemory(),
        ArmCogniCore(arm_name="Arm_3_CogniCoreRetrieval", use_reflection=False),
        ArmCogniCore(arm_name="Arm_4_CogniCoreFull", use_reflection=True)
    ]
    
    domains = [
        ("software_debugging", PythonSandboxEnv, generate_python_tasks),
        ("deployment_infrastructure", DockerLinterEnv, generate_docker_tasks)
    ]

    for domain_name, EnvClass, generator_fn in domains:
        print(f"\n=== Domain: {domain_name} ===")
        
        # We pre-generate the tasks for determinism across arms
        tasks = generator_fn(num_episodes=args.episodes, seed=42)
        
        for arm in arms:
            print(f"\n--- Arm: {arm.arm_name} ---")
            
            for run_id in range(1, args.runs + 1):
                arm.reset_run()
                env = EnvClass()
                
                print(f"  Run {run_id}/{args.runs} ", end="", flush=True)
                for episode_idx, task in enumerate(tasks):
                    # For different runs, we would ideally use different seeds for tasks,
                    # but for now we reuse the same sequence of 20 tasks per run so we can average out the LLM stochasticity.
                    result = arm.run_episode(env, task, max_turns=5)
                    
                    tracker.log_episode(
                        run_id=run_id,
                        arm_name=arm.arm_name,
                        domain=domain_name,
                        episode_id=episode_idx + 1,
                        concept=task.get("concept_type", "unknown"),
                        success=result["success"],
                        retries=result["retries"],
                        first_action_acc=result["first_action_accuracy"],
                        rfr=result["repeated_failure_rate"],
                        tokens_prompt=result["tokens_prompt"],
                        tokens_completion=result["tokens_completion"],
                        cost_usd=result["cost_usd"],
                        tts_sec=0.0 # Time tracking omitted for brevity, can be added
                    )
                    
                    if result["success"]:
                        print(".", end="", flush=True)
                    else:
                        print("x", end="", flush=True)
                        
                print(" (Done)")

if __name__ == "__main__":
    main()
