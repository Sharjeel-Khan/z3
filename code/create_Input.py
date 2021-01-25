import random 
N = 1000

seed = "sat.random_seed="

gc_increment = "sat.gc.increment="
gc_initial = "sat.gc.initial="
gc_k = "sat.gc.k="

if __name__ == '__main__':
    with open("inputs.txt","w") as file:
        for i in range(N):
            random_seed = random.randint(0, 100)
            random_initial = random.randint(50, 200000)
            random_increment = random.randint(50, random_initial)
            random_k = random.randint(1, 10)

            params_list = [seed+str(random_seed),gc_increment+str(random_increment),gc_initial+str(random_initial),gc_k+str(random_k)]
            params = " ".join(params_list)
            
            file.write(params+"\n")
