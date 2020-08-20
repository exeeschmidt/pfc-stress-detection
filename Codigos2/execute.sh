#!/bin/bash
#SBATCH --job-name=Stress2Exp # nombre para identificar el trabajo. Por defecto es el nombre del script
#SBATCH --ntasks=3 # cantidad de cores pedidos
#SBATCH --tasks-per-node=3 # cantidad de cores por nodo, para que distribuya entre varios nodos
##SBATCH --nodes=1 # solicita P=1 nodos completos. Tener en cuenta lanzar la cantidad adecuada de procesos para que el nodo no quede subtulizado.
#SBATCH --output=trabajo-%j-salida.txt # la salida y error estandar van a este archivo. Si no es especifca es slurm-%j.out (donde %j es el Job ID)
#SBATCH --error=trabajo-%j-error.txt # si se especifica, la salida de error va por separado a este archivo
#SBATCH --time=15-0
#SBATCH --mem=32G
#SBATCH --mail-user=tonu17695@gmail.com --mail-type=end

# aqui comienzan los comandos
module load python3
python3 main.py