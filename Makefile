openai_ceval:
	python reLLMs/eval.py -m task=ceval task.name=high_school_biology model=openai model.name=gpt-3.5-turbo

llama7b_ceval:
	python reLLMs/eval.py -m +exp=llama7b_ceval task.name=computer_network,operating_system

llama65b_ceval:
	python reLLMs/eval.py -m +exp=llama65b_ceval task.name=computer_network,operating_system
