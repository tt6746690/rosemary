function speek {
    if [[ "$1" == "-i" ]]; then
        # Handle indexing functionality
        index=$2
        # Get a list of running job IDs using squeue, filtering for running jobs
        running_jobs=($(squeue --me -h -o "%i" --sort=t,i))
        
        if [[ $index -lt 0 ]]; then
            # Convert negative index to positive
            total_jobs=${#running_jobs[@]}
            index=$((total_jobs + index))
        fi

        if [[ $index -ge 1 && $index -le ${#running_jobs[@]} ]]; then
            job_id=${running_jobs[$index]}
            log_file="/fsx/wpq/.slurm_log/$job_id.out"
            if [[ -f "$log_file" ]]; then
                cat "$log_file"
                echo
                echo "speek $job_id"
            else
                echo "Log file $log_file for job $job_id does not exist."
            fi
        else
            echo "Index out of range"
            return 1
        fi
    elif [[ $# -eq 1 ]]; then
        job_id=$1
        log_file="/fsx/wpq/.slurm_log/$job_id.out"
        if [[ -f "$log_file" ]]; then
            cat "$log_file"
        else
            echo "$job_id finished or log file does not exist."
        fi
    else
        echo "Usage: speek <job_id> or speek -i <index>"
        return 1
    fi
}