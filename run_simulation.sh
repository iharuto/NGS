#!/bin/bash

# Run the neuron growth simulation
echo "Running neuron growth simulation..."
python axon_growth_sim.py

# Update README.md with gallery of latest final results
echo "Updating README.md..."

# Create README.md header
cat > README.md << 'EOF'
# Neuron Growth Simulation

Simulate axonal growth in a 2D cellular-automaton world.

## Gallery

EOF

# Get list of PNG files in final_results directory, sorted by filename (newest first)
if [ -d "final_results" ] && [ "$(ls -A final_results/*.png 2>/dev/null)" ]; then
    image_files=($(ls -1 final_results/*.png | sort -r | head -6))

    echo '<table>' >> README.md

    for i in "${!image_files[@]}"; do
        if (( i % 3 == 0 )); then
            echo '  <tr>' >> README.md
        fi

        file="${image_files[$i]}"
        filename=$(basename "$file")
        echo "    <td><img src=\"$file\" width=\"200\"></td>" >> README.md

        if (( i % 3 == 2 )); then
            echo '  </tr>' >> README.md
        fi
    done

    # 最後の行が閉じられていない場合に閉じる
    if (( ${#image_files[@]} % 3 != 0 )); then
        echo '  </tr>' >> README.md
    fi

    echo '</table>' >> README.md
else
    echo "No final results available yet." >> README.md
fi

# Add footer
cat >> README.md << 'EOF'

## Usage

Run the simulation and update results:
```bash
# conda create -n axon python=3.11 # for initial setup
# pip install -r requirements.txt # for initial setup
conda activate axon
./run_simulation.sh
```

## Environment

- conda environment: axon
- Required packages: see requirements.txt
EOF

echo "README.md updated successfully."

# Commit and push using existing script
echo "Committing and pushing results..."
./commit_and_push.sh

echo "Process completed!"