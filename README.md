# Neuron Growth Simulation

Simulate axonal growth in a 2D cellular-automaton world. The repulsion area (blue-colored regions) are stochastically generated based on Mac desktop app locations, so that you can enjoy how neuron seeds interactively grow.


## Gallery

<table>
  <tr>
    <td><img src="final_results/250722.png" width="200"></td>
    <td><img src="final_results/250721.png" width="200"></td>
    <td><img src="final_results/250720.png" width="200"></td>
  </tr>
  <tr>
    <td><img src="final_results/250719.png" width="200"></td>
    <td><img src="final_results/250718.png" width="200"></td>
    <td><img src="final_results/250717.png" width="200"></td>
  </tr>
</table>

## Usage

Run the simulation and update results:
```bash
# conda create -n axon python=3.11 # for initial setup
# pip install -r requirements.txt # for initial setup
conda activate axon
./run_simulation.sh
```

## Environment

- OS environment: M3 macOS 13.5
- conda environment: axon
- Required packages: see requirements.txt
