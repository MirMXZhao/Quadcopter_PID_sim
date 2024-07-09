"""
simulates first order ODE using runge-kutta method
"""
import matplotlib.pyplot as plt

num_rounds = 50

simulated_x = [None]*num_rounds

def func(x, A, ):
   """
   x^{dot} = func(x,t)
   """
   return 2*x - t
   pass


def simulate_runge_kutta(fn, x0, t0, step_size):
   simulated_x = [None]*num_rounds
   simulated_x[0] = x0
   cur_t =t0
   for i in range(1, num_rounds):
       k1 = fn(simulated_x[i-1], cur_t)
       k2 = fn(simulated_x[i-1] + step_size*k1/2, cur_t + step_size/2)
       k3 = fn(simulated_x[i-1] + step_size*k2/2, cur_t + step_size/2)
       k4 = fn(simulated_x[i-1] + step_size*k3, cur_t + step_size)
       simulated_x[i] = simulated_x[i-1] + step_size/6*(k1 + 2*k2 + 2*k3 + k4)
       cur_t = cur_t + step_size
   return simulated_x

def simulate_eulers(fn, x0, t0, step_size):
   simulated_x = [None]*num_rounds
   simulated_x[0] = x0
   cur_t = t0
   for i in range(1, num_rounds):
      simulated_x[i] = simulated_x[i-1] + step_size*fn(simulated_x[i-1], cur_t)
      cur_t = cur_t + step_size
   return simulated_x

def plot(values):
   plt.figure()
   for i in range(len(values)):
      plt.plot(range(0, num_rounds), values[i])
   plt.show()
   pass


if __name__ == "__main__":
   x0 = 1/2
   t0 = 0
   h = 0.05
   fn = func
   simulated_rk = simulate_runge_kutta(fn, x0, t0, h)
   simulated_eu = simulate_eulers(fn, x0, t0, h)
   print(simulated_rk)
   print(simulated_eu)
   plot([simulated_rk, simulated_eu])
