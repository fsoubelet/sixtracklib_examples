
# In[1]:


import numpy as np

import pysixtrack
import sixtracklib


# In[2]:


def get_sixtracklib_particle_set(init_physical_coordinates: np.ndarray, p0c_ev: float):
    n_particles = init_physical_coordinates.shape[0]
    ps = sixtracklib.ParticlesSet()
    p = ps.Particles(num_particles=n_particles)

    for i_part in range(n_particles):
        part = pysixtrack.Particles(p0c=p0c_ev)

        part.x = init_physical_coordinates[i_part, 0]
        part.px = init_physical_coordinates[i_part, 1]
        part.y = init_physical_coordinates[i_part, 2]
        part.py = init_physical_coordinates[i_part, 3]
        part.tau = init_physical_coordinates[i_part, 4]
        part.ptau = init_physical_coordinates[i_part, 5]

        part.partid = i_part
        part.state = 1
        part.elemid = 0
        part.turn = 0

        p.from_pysixtrack(part, i_part)

    return ps


# In[3]:


from cpymad.madx import Madx

madx = Madx(stdout=False)
madx.call("lhc_colin_track.seq")
madx.use("lhcb1")  # if not called, no beam present in sequence???
_ = madx.twiss()  # if not called, cavities don't get a frequency???


# In[4]:


# Line and sixtracklib Elements
line = pysixtrack.Line.from_madx_sequence(madx.sequence["lhcb1"])

# Only the coordinates at the end of tracking. To keep coordinates at each
# turn, give "num_stores=nturns" when creating the BeamMonitor element
_ = line.append_element(pysixtrack.elements.BeamMonitor(num_stores=1, is_rolling=True), "turn_monitor")
elements = sixtracklib.Elements.from_line(line)


# In[5]:


# CO, Linear_OTM and W
closed_orbit, linear_otm = line.find_closed_orbit_and_linear_OTM(
    p0c=madx.sequence["lhcb1"].beam.pc * 1e9, longitudinal_coordinate="tau"
)
W, invW, R = line.linear_normal_form(linear_otm)


# ## Create distribution
# ---

# In[6]:


n_parts = 1_000
energy = 6500
geometric_emittance = 3.75e-6 / (energy / 0.938);


# In[7]:


# Get normalized coordinates yolo style
x_normalized = list(np.random.normal(0, np.sqrt(geometric_emittance), n_parts))
px_normalized = list(np.random.normal(0, np.sqrt(geometric_emittance), n_parts))
y_normalized = list(np.random.normal(0, np.sqrt(geometric_emittance), n_parts))
py_normalized = list(np.random.normal(0, np.sqrt(geometric_emittance), n_parts))
tau_normalized = list(np.random.normal(0, 125 * np.sqrt(geometric_emittance), n_parts))
ptau_normalized = list(np.random.normal(0, 125 * np.sqrt(geometric_emittance), n_parts))
normalized_coordinates = np.array(
    [x_normalized, px_normalized, y_normalized, py_normalized, tau_normalized, ptau_normalized]
)


# In[8]:


# Go from normalized to physical and center on closed orbit
initial_coordinates = (W @ normalized_coordinates).T
initial_coordinates += closed_orbit

# make sure that stdev of tau (~ bunch length) is correct, should be around 0.08 meters
print(initial_coordinates[:, 4].std())
# the error is stdev / sqrt(2 * n_part)
print(initial_coordinates[:, 4].std() / np.sqrt(2 * n_parts))


# ## Prepare Job & Track 
# ---

# In[9]:


particle_set = get_sixtracklib_particle_set(
    init_physical_coordinates=initial_coordinates, p0c_ev=madx.sequence["lhcb1"].beam.pc * 1e9
)
particle_set.to_file("here.particleset")


# In[10]:


# Or from dumped file
# particle_set = sixtracklib.ParticlesSet.fromfile("here.particleset")


# In[11]:


sixtracklib.TrackJob.print_nodes("opencl")


# In[ ]:


# For GPU use, specify GPU device_id from information given by clinfo in the cell above
job = sixtracklib.TrackJob(elements, particle_set, device="opencl:0.2")
job.track_until(100)
job.collect()  # transfer data back from GPU, fine to call if CPU only


# In[ ]:


final_output = {
    "x": job.output.particles[0].x,
    "px": job.output.particles[0].px,
    "y": job.output.particles[0].y,
    "py": job.output.particles[0].py,
    "zeta": job.output.particles[0].zeta,
    "delta": job.output.particles[0].delta,
    "at_turn": job.output.particles[0].at_turn,
}


# In[ ]:


final_output


# In[ ]:


# trick to get tau and ptau: define a Particle with these results in pysixtrack and get the results from there
p = pysixtrack.Particles(p0c=madx.sequence["lhcb1"].beam.pc * 1e9)
p.zeta = final_output["zeta"]
p.delta = final_output["delta"]
p.tau


# ---
