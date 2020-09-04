"""This script is used to simulate the impact of an interruption of
feedstock feed-in on biogas production"""

import numpy as np
import bipmo.biogas_plant_model
import matplotlib.pyplot as plt

# List of the biogas production potential in m3 of biogas per kg of feedstock [5]
potential_sustrate_list = [
    # for corn silage
    0.203775,
    # for sugar beet silage
    0.1449
    ]

# Selection of the feedstock type according to the scenario chosen
potential_sustrate = potential_sustrate_list[0]

# Load the scenario
scenario_name = 'scenario_1'
bg = bipmo.biogas_plant_model.BiogasModel(scenario_name)

# Feed-in regime
u = np.array([0])
u_int = np.array([0])

for i in range(len(bg.control_matrix.columns)-2):
    u = np.vstack((u, np.array([0])))
    u_int = np.vstack((u_int, np.array([0])))

u = np.vstack((u, np.array([1])))  # Maximum feed-in regime
u_int = np.vstack((u_int, np.array([0])))  # Zero feed-in regime



# Initial state maximal production
x0 = np.array([[potential_sustrate], [2000]])

# Duration of the simulation in hours
length = 48

# Instantiate the variables
x = x0
y1 = []  # biogas production level list in %
x1 = bg.state_matrix.dot(x) + bg.control_matrix.dot(u)
x = x1
total_prod = [float(x1[0][scenario_name + '_prod_biogas_m3_s-1'])*bg.timestep_interval.seconds]   # Total production of the digester

# Time slot for the interruption
deb_interruption = 6
end_interruption = 12

# Feed-in regime profile list in percentage (used for plot)
u_plot = []

# Simulation before interruption
for i in range(0, deb_interruption+1):
    total_prod.append(total_prod[-1])
    x = bg.state_matrix.dot(x)+bg.control_matrix.dot(u)
    y = bg.state_output_matrix.dot(x) + bg.control_output_matrix.dot(u)
    y1.append(x[0][scenario_name + '_prod_biogas_m3_s-1']/potential_sustrate*100)
    total_prod[-1] = total_prod[-1] + x[0][scenario_name + '_prod_biogas_m3_s-1']*bg.timestep_interval.seconds
    u_plot.append(100)

# Simulation during interruption
for i in range(deb_interruption+1, end_interruption+1):
    total_prod.append(total_prod[-1])
    x = bg.state_matrix.dot(x) + bg.control_matrix.dot(u_int)
    y = bg.state_output_matrix.dot(x) + bg.control_output_matrix.dot(u)
    y1.append(x[0][scenario_name + '_prod_biogas_m3_s-1']/potential_sustrate*100)
    total_prod[-1] = total_prod[-1] + x[0][scenario_name + '_prod_biogas_m3_s-1']*bg.timestep_interval.seconds
    u_plot.append(0)


# Simulation after restart dof feed-in
for i in range(end_interruption+1, length+1):
    total_prod.append(total_prod[-1])
    x = bg.state_matrix.dot(x)+bg.control_matrix.dot(u)
    y = bg.state_output_matrix.dot(x) + bg.control_output_matrix.dot(u)
    y1.append(x[0][scenario_name + '_prod_biogas_m3_s-1']/potential_sustrate*100)
    total_prod[-1] = total_prod[-1] + x[0][scenario_name + '_prod_biogas_m3_s-1']*bg.timestep_interval.seconds
    u_plot.append(100)


plt.plot(y1, linewidth=2, color='k')
plt.plot(u_plot, linewidth=2, color='k', linestyle='--', drawstyle='steps-pre')
plt.legend(["Biogas production (%)", "Feed-in regime (%)"])
plt.xticks(([0, 6, 12, 18, 24, 30, 36, 42, 48]))
plt.yticks([0, 25, 50, 75, 100])
plt.axhline(y=75, linestyle='--', color='k', linewidth=1)
plt.axhline(y=50, linestyle='--', color='k', linewidth=1)
plt.axhline(y=25, linestyle='--', color='k', linewidth=1)
plt.ylim(0, 100)
plt.xlabel('Time (h)')

plt.show()
plt.close()

plt.plot(total_prod)
plt.xlabel('Time (h)')
plt.ylabel('Total biogas production (m3)')
plt.show()
plt.close()
