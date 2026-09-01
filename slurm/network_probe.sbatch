#!/usr/bin/env bash
#SBATCH --job-name=seamless-net-probe
#SBATCH --account=simurgh
#SBATCH --partition=simurgh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/network_probe_%j.out
#SBATCH --error=slurm/logs/network_probe_%j.err

set -u

date --iso-8601=seconds
hostname --fqdn

for host in pypi.org huggingface.co repo.anaconda.com; do
    getent ahostsv4 "${host}" | head -n 1 || true
    curl --head --location --max-time 15 --silent --show-error \
        --output /dev/null --write-out "${host} http=%{http_code}\n" \
        "https://${host}/" || true
done
