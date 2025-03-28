import os
import time
import sys
import json
import csv
import random
from dotenv import load_dotenv
from prettytable import PrettyTable
from multiprocessing import Pool, cpu_count
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()
random.seed(42)

debug = False


def get_client(llm_type, llm_size):
    if llm_size == "small":
        endpoint = os.getenv(f"{llm_type.upper()}_SMALL_ENDPOINT")
        api_key = os.getenv(f"{llm_type.upper()}_SMALL_API_KEY")
    elif llm_size == "large":
        endpoint = os.getenv(f"{llm_type.upper()}_LARGE_ENDPOINT")
        api_key = os.getenv(f"{llm_type.upper()}_LARGE_API_KEY")
    else:
        raise ValueError("Invalid llm_size. Choose from 'small' or 'large'.")

    if not endpoint or not api_key:
        raise Exception("Endpoint and API key must be provided in the .env file")

    if llm_type in ["llama", "mistral"]:
        credential = AzureKeyCredential(api_key)
        return ChatCompletionsClient(endpoint=endpoint, credential=credential)
    elif llm_type == "gpt":
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-05-01-preview",
        )
    else:
        raise ValueError("Invalid llm_type. Choose from 'llama', 'mistral', or 'gpt'.")


def call_llm(idx, content, prompt, temperature, top_p, llm_type="mistral", llm_size="small", _input_cost=0.0, _output_cost=0.0):
    client = get_client(llm_type, llm_size)
    payload = {
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": 0,
    }

    if llm_type == "gpt" and llm_size == "small":
        deployment_name = os.getenv("GPT_SMALL_DEPLOYMENT_NAME")
    elif llm_type == "gpt" and llm_size == "large":
        deployment_name = os.getenv("GPT_LARGE_DEPLOYMENT_NAME")
    else:
        pass


    retry = 5
    answer = None
    cost = 0

    while retry > 0:
        try:
            if llm_type == "gpt":
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=payload["messages"],
                    max_tokens=800,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                response = json.loads(response.model_dump_json(indent=2))

                cost = round((response["usage"]["prompt_tokens"] / 1000) * _input_cost, 4) + round(
                   (response["usage"]["completion_tokens"] / 1000) * _output_cost, 4)

                answer = response['choices'][0]['message']['content']

            else:
                response = client.complete(payload)
                answer = response.choices[0].message.content
                cost = response.usage.prompt_tokens * _input_cost + response.usage.completion_tokens * _output_cost
            break
        except Exception as e:
            print(f"Error: {e}. Retrying... ({retry} attempts left)", flush=True)
            time.sleep(60)
        retry -= 1
    return (idx, str(answer) if answer else "ERROR PROCESSING REQUEST.", cost)


def call_llm_fitness(idx, content, tag, prompt, temperature, top_p, llm_type="mistral", llm_size="small", _input_cost=0.0, _output_cost=0.0):
    client = get_client(llm_type, llm_size)


    payload = {
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": 0,
    }
    retry = 5
    cost = 0
    answer = None

    if llm_type == "gpt" and llm_size == "small":
        deployment_name = os.getenv("GPT_SMALL_DEPLOYMENT_NAME")
    elif llm_type == "gpt" and llm_size == "large":
        deployment_name = os.getenv("GPT_LARGE_DEPLOYMENT_NAME")
    else:
        pass

    while retry > 0:
        try:
            if llm_type == "gpt":
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=payload["messages"],
                    max_tokens=800,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                response = json.loads(response.model_dump_json(indent=2))
                answer = response['choices'][0]['message']['content']
                cost = round((response["usage"]["prompt_tokens"] / 1000) * _input_cost, 4) + round(
                    (response["usage"]["completion_tokens"] / 1000) * _output_cost, 4)
            else:
                response = client.complete(payload)
                answer = response.choices[0].message.content
                cost = response.usage.prompt_tokens * _input_cost + response.usage.completion_tokens * _output_cost

            break
        except Exception as e:
            print(f"Error: {e}. Retrying... ({retry} attempts left)", flush=True)
            time.sleep(60)
        retry -= 1
    return (idx, str(answer) if answer else "ERROR PROCESSING REQUEST.", cost, tag)


#Fitness function
def fitness_function(prompt,mode="population"):
    global _data, _system_fitness, _fitness, _system_message, evaluation_pop_size, _FitnessMem, _input_cost,_output_cost
    hit = 0
    ind_cost = 0.0

    data = _data

    total = len(data)
    Hit = {}
    Match = {}
    def collect(result):
        idx,prompt,cost = result
        if prompt.startswith("OUTPUT:"):
            prompt = prompt.replace("OUTPUT: ","")
        Hit[idx]=(prompt,cost)

    pool = Pool(cpu_count()-1)
    for idx,values in enumerate(data):
        text,tag = values
        if debug==False:
            pool.apply_async(call_llm,args=(idx,_system_message,prompt+"\n\nINPUT: "+text+"\n",0.7,0.95, llm_type, llm_size, _input_cost, _output_cost),callback=collect)
        else:
            collect(call_llm(idx,_system_message,prompt+"\n\nINPUT: "+text+"\n",0.7,0.95, llm_type, llm_size, _input_cost, _output_cost))
    pool.close()
    pool.join()

    def collect_fitness(result):
        global _FitnessMem
        idx,answer,cost,tag = result
        if answer.startswith("OUTPUT:"):
            answer = answer.replace("OUTPUT: ","")
        if not (answer.strip(),tag.strip().lower()) in _FitnessMem:
            _FitnessMem[(answer.strip(),tag.strip().lower())] = answer.strip()
        Match[idx]=(answer,cost)

    pool = Pool(cpu_count()-1)
    for idx in Hit:
        tag = data[idx][-1]
        answer,cost = Hit[idx]
        ind_cost += cost
        if (answer.strip(),tag.strip().lower()) in _FitnessMem:
            collect_fitness((idx,_FitnessMem[(answer.strip(),tag.strip().lower())],0.0,tag.strip().lower()))
        elif tag.strip().lower() in answer.strip().lower():
            collect_fitness((idx,'yes',0.0,tag.strip().lower()))
        else:
            if debug==False:
                pool.apply_async(call_llm_fitness,args=(idx,_system_fitness,tag.strip(),_fitness.replace("XXXXX",tag.strip()).replace("YYYYY",answer),0.7,0.95, llm_type, llm_size, _input_cost, _output_cost),callback=collect_fitness)
            else:
                collect_fitness(call_llm_fitness(idx,_system_fitness,tag.strip(),_fitness.replace("XXXXX",tag.strip()).replace("YYYYY",answer),0.7,0.95, llm_type, llm_size, _input_cost, _output_cost))
    pool.close()
    pool.join()
    for idx in Match:
        tag = data[idx][-1]
        answer,cost = Hit[idx]
        match,cost2 = Match[idx]
        ind_cost += cost2
        if 'yes' in match.lower().strip():
            hit += 1
            if cost2 > 0.0:
                print("GOLDEN:",tag,flush=True)
                print("ANSWER:",answer,flush=True)
                print("MATCH: ",match,flush=True)

    if len(Match) != total:
        print("WARNING: Some queries were not processed. Processed results",len(Match),"is different than total queries",total,flush=True)

    try:
        resp = (float(hit)/float(len(Match)),ind_cost)
    except:
        resp = (0.0,0.0)
    return resp

# Create the initial population
def create_initial_population(size):
    global _system_message, _init_prompt, _input_cost, _output_cost, llm_type, llm_size
    population = []
    Prompts    = {}
    init_cost  = 0.0

    def collect(result):
        idx,prompt,cost = result
        print("--INITIAL PROMPT",idx+1,"--\n",prompt,"\n------------------------------------------------")
        Prompts[idx] = (prompt,cost)

    pool=Pool(cpu_count()-1)
    print("_____________________________________________________")
    print("Generating initial population...")
    for i in range(size):
        if debug==False:
            pool.apply_async(call_llm,args=(i,_system_message,_init_prompt,1.0,0.90, llm_type, llm_size, _input_cost, _output_cost),callback=collect)
        else:
            collect(call_llm(i,_system_message,_init_prompt,1.0,0.90, llm_type, llm_size, _input_cost, _output_cost))

    pool.close()
    pool.join()

    for idx in Prompts:
        prompt,cost = Prompts[idx]
        init_cost += cost
        population.append(prompt)
    return (population,init_cost)

# Selection function using tournament selection
def run_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# Crossover function
def run_crossover(parent1, parent2):
    global _system_message, _crossover_prompt
    print("_____________________________________________________")
    print("Crossing over 2 individuals...")
    print("INPUT PROMPT 1:\n",parent1,"\nINPUT PROMPT 2:\n",parent2)
    print(".....................................................")
    idx,child1,cost1 = call_llm(0,_system_message,_crossover_prompt+"\nPROMPT 1:\n"+parent1+"\n\nPROMPT 2:\n"+parent2+"\n",1.0,0.90, llm_type, llm_size, _input_cost, _output_cost)
    print("CROSSED PROMPT 1:\n",child1)
    print("Crossover cost:",cost1)
    print(".....................................................")
    idx,child2,cost2 = call_llm(0,_system_message,_crossover_prompt+"\nPROMPT 1:\n"+parent2+"\n\nPROMPT 2:\n"+parent1+"\n",1.0,0.90, llm_type, llm_size, _input_cost, _output_cost)
    print("CROSSED PROMPT 2:\n",child2)
    print("Crossover cost:",cost2)
    print("_____________________________________________________")
    return child1, child2, cost1+cost2

# Mutation function
def run_mutation(individual, mutation_rate):
    global _mutation_prompt
    cost = 0.0
    if random.random() < mutation_rate:
        print("_____________________________________________________")
        print("Mutating 1 individual...")
        print("INPUT PROMPT:\n",individual)
        print(".....................................................")
        idx,individual,cost = call_llm(0,_system_message,_mutation_prompt+"\nPROMPT:\n"+individual+"\n",1.0,0.90,llm_type, llm_size, _input_cost, _output_cost)
        print("MUTATED PROMPT:\n",individual)
        print("_____________________________________________________")
    return (individual,cost)

# Main genetic algorithm function
def genetic_algorithm(population_size, generations, mutation_rate, evaluation_pop_size):
    global _data, _FitnessMem, _input_cost, _output_cost
    population,init_cost = create_initial_population(population_size)

    # Prepare for plotting
    best_performers = []
    all_populations = []
    overall_pcalls  = 0
    overall_ncalls  = 0
    overall_cost    = init_cost
    # Prepare for table
    table = PrettyTable()
    table.field_names = ["Generation", "Best Fitness", "Best Individual", "Avg Fitness", "Number of LLM Prompts", "Number LLM Calls", "Accumulated Cost"]
    for generation in range(generations):
        gen_cost  = 0.0
        gen_max   = 0.0
        gen_avg   = 0.0
        fitnesses = []

        print("-- GEN",generation,"Evaluating individuals in population...")
        for ii,ind in enumerate(population):
            print("........................\n  Individual",ii+1,"\nPROMPT",ind,"\n........................")
            fitness,ind_cost = fitness_function(ind,"population")
            gen_cost += ind_cost
            gen_avg  += fitness
            print("\tfitness:",fitness)
            print("\tEval cost:",ind_cost)
            fitnesses.append(fitness)

            print("-- PROMPT EVAL",ii+1,"\nFITNESS",fitness,"\n------------------------------------------------")

        overall_pcalls += population_size
        overall_ncalls += population_size*evaluation_pop_size
        overall_cost   += gen_cost
        gen_max         = max(fitnesses)
        gen_avg        /= len(population)

        # Store the best performer of the current generation
        best_index      = fitnesses.index(gen_max)
        best_individual = population[best_index]
        best_fitness = fitnesses[best_index]
        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        print("-----------------STATUS--------------------")
        table.add_row([generation,best_fitness,best_index,gen_avg,overall_pcalls,overall_ncalls,overall_cost])
        print(table)
        print("BEST INDIVIDUAL IN "+str(generation)+"TH GENERATION:")
        print(best_individual)

        print()
        print("_____________________________________________________")
        print("Selecting next generation...")
        population = run_selection(population, fitnesses)

        next_population = []
        crossover_cost  = 0.0
        nmutation       = 0
        for i in range(0, len(population)-1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            child1, child2, cost = run_crossover(parent1, parent2)

            crossover_cost += cost
            child1,cost = run_mutation(child1, mutation_rate)
            if cost > 0.0:
                nmutation += 1
                crossover_cost += cost
            child2,cost = run_mutation(child2, mutation_rate)
            if cost > 0.0:
                nmutation += 1
                crossover_cost += cost

            next_population.append(child1)
            next_population.append(child2)

        overall_pcalls += population_size + nmutation
        overall_ncalls += population_size + nmutation #population_size = number of crossovers
        overall_cost   += crossover_cost

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population

    # Plot the population of one generation (last generation)
    final_population = all_populations[-1] if len(all_populations) > 0 else population
    overall_pcalls   = overall_pcalls if len(all_populations) > 0 else population_size
    final_fitnesses = []
    final_avg       = 0.0
    final_cost      = 0.0
    print("Evaluating individuals in final population...")
    for ii,ind in enumerate(final_population):
        print("........................\n  Individual",ii+1,"\n",ind,"\n........................")
        fitness,cost = fitness_function(ind,"population")
        final_fitnesses.append(fitness)
        final_avg  += fitness
        final_cost += cost
        print("-- FINAL GEN -- PROMPT EVAL",ii,"--\nPROMPT:",ind,"\nFITNESS",fitness,"\n------------------------------------------------")

    final_max       = max(final_fitnesses)
    final_avg      /= len(final_population)
    best_index      = final_fitnesses.index(final_max)
    best_individual = final_population[best_index]
    best_fitness = final_fitnesses[best_index]
    best_performers.append((best_individual, best_fitness))
    overall_ncalls += len(final_population)*evaluation_pop_size
    overall_cost   += final_cost

    print("-----------------FINAL STATUS--------------------")
    table.add_row([generations,best_fitness,best_index,final_avg,overall_pcalls,overall_ncalls,overall_cost])
    print(table)

    print("BEST INDIVIDUAL IN FINAL GENERATION:")
    print(best_individual)

    max_fit = max(final_fitnesses)
    final_index = final_fitnesses.index(max_fit)
    return population[final_index], best_fitness, overall_cost #final_fitnesses[final_index], overall_cost

if __name__ == "__main__":
    _t0 = time.time()
    _dataset , _nexamples,  llm_type, llm_size = sys.argv[-4:]
    tcol = 0
    lcol = 1

    # Parameters for the genetic algorithm
    population_size = 20
    generations     = 20
    mutation_rate   = 0.1

    examples_size   = int(_nexamples)
    evaluation_pop_size = 500


    _examples = []
    _data = []
    with open(_dataset) as fin:
        csvreader = list(csv.reader(fin, delimiter=','))
        random.shuffle(csvreader)
        for iline,line in enumerate(csvreader):
            if iline == 0:
                continue
            if iline <= examples_size:
                _examples.append(["INPUT: "+line[tcol].strip(),"OUTPUT: "+line[lcol].strip()])
            elif iline <= examples_size+evaluation_pop_size:
                _data.append([line[tcol].strip(),line[lcol].strip()])
            else:
                break


    _FitnessMem       = {}

    _Cost = {
        'GPT4-turbo': (0.005, 0.015),
        'GPT4o-mini': (0.00015, 0.00060),
        'Mistral NeMo': (0.0000003, 0.0000003),
        'Llama 3.2 11B instruct': (0.00000037, 0.00000037),
        'Llama 3.3 70B': (0.00000071, 0.00000071),
        'Mistral Large 2': (0.000002, 0.000006)
    }


    llm_name = None
    if llm_type == "mistral" and llm_size == "small":
        llm_name = os.getenv("LLM_MISTRAL_SMALL")
    elif llm_type == "mistral" and llm_size == "large":
        llm_name = os.getenv("LLM_MISTRAL_LARGE")
    elif llm_type == "gpt" and llm_size == "small":
        llm_name = os.getenv("LLM_GPT_SMALL")
    elif llm_type == "gpt" and llm_size == "large":
        llm_name = os.getenv("LLM_GPT_LARGE")
    elif llm_type == "llama" and llm_size == "small":
        llm_name = os.getenv("LLM_LLAMA_SMALL")
    elif llm_type == "llama" and llm_size == "large":
        llm_name = os.getenv("LLM_LLAMA_LARGE")
    else:
        raise ValueError("Invalid llm_type or llm_size")

    _input_cost, _output_cost = _Cost.get(llm_name, (0.0, 0.0))

    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')


    print('STARTING GA WITH PARAMETERS:')
    print('population_size:',population_size)
    print('generations:',generations)
    print('mutation_rate:',mutation_rate)
    print('evaluation_pop_size:',evaluation_pop_size)

    print('Running Dataset:',_dataset)
    print(f'Number of data samples used during initialization for context learning: {examples_size}')
    print(f'LLM Type: {llm_type}')
    print(f'LLM Size: {llm_size}')
    print(f'LLM Name: {llm_name}')
    print(f'Input Cost: {_input_cost} per token')
    print(f'Output Cost: {_output_cost} per token')

    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')
    print(f'################################################################################################')


    _llm              = str(llm_name)
    _system_fitness   = "You are an AI that validates automated answers against ground truth. You only answer 'yes' or 'no' with no extra comments, notes, or explanations as the user will use your answer to integrate with another downstream system."
    _fitness          = "My ground-truth is 'XXXXX'. Does the automated output 'YYYYY' linguistically, symbolically, conceptually, or fundamentally match my ground-truth?"

    _system_message   = "You are an AI that helps people solve problems. Avoid comments outside the proposed response as the user will use your answer to integrate with another downstream system."
    _init_prompt      = "Create a "+_llm+" prompt that solves the underlying problem exemplified by the following examples:\n"
    for ex_input,ex_output in _examples:
        _init_prompt  = _init_prompt + ex_input + "\n" + ex_output + "\n\n"
    _crossover_prompt = "Given the following drafts for two prompts that aim to solve the same particular problem, create a better prompt using only ideas from them."
    _mutation_prompt  = "Given the following draft for a prompt that aims to solve a particular problem, create a better prompt with using ideas from it."
    print(_init_prompt)
    #Run the GA
    best_solution, best_fitness, overall_cost = genetic_algorithm(population_size, generations, mutation_rate, evaluation_pop_size)
    print("======================================================================")
    print("BEST SOLUTION:")
    print(best_solution)
    print()
    print("BEST FITNESS:",best_fitness)
    print("OVERALL COST:",overall_cost)
    print("OVERALL TIME:",(time.time()-_t0)/60.0,"mins.")
