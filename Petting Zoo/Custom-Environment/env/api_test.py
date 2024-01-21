from pettingzoo.test import parallel_api_test
import custom_environment

if __name__ == "__main__":
    env = custom_environment.CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)