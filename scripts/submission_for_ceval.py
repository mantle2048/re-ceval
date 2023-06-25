import fire
import json
import warnings
import pandas as pd
from pathlib import Path

TASK_NAMES = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician"
]


def format_exps(model: str='llama-7b-hf'):
    exp_dir = Path('../exps/ceval') / model
    submission_dict = {}
    for task in TASK_NAMES:
        run_paths = list(exp_dir.rglob(f'{model}_{task}/*{task}*'))
        if not run_paths:
            warnings.warn(f"Not find any runs of task:{task} in {exp_dir}, just skip")
            continue
        if len(run_paths) > 1:
            warnings.warn(f"Multiple runs of task:{task} in {exp_dir} founded, only extract the first run")
        run = run_paths[0]
        df = pd.read_csv(run/'progress.csv').fillna('')
        submission_dict[task] = \
            df.set_index('id')['answer_model'].to_dict()
    with open('submission.json', 'w') as file:
        json.dump(submission_dict, file)


if __name__ == '__main__':
    fire.Fire(format_exps)
