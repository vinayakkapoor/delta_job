import json

def find_new_jobs(file1_path, file2_path, output_path=None):
    """
    Compare two JSON files and find new job titles in the second file
    that don't exist in the first file for each company
    """
    # Load data from both files
    with open(file1_path, 'r') as f:
        file1_data = json.load(f)
    
    with open(file2_path, 'r') as f:
        file2_data = json.load(f)
    
    result = {}
    
    # Compare jobs for each company in the second file
    for company, jobs_in_file2 in file2_data.items():
        # Get jobs from first file or empty list if company doesn't exist
        jobs_in_file1 = file1_data.get(company, [])
        
        # Find jobs in file2 that aren't in file1 (preserving order and duplicates)
        new_jobs = [job for job in jobs_in_file2 if job not in jobs_in_file1]
        
        if new_jobs:  # Only add companies with new jobs
            result[company] = new_jobs
    
    if output_path is not None:
        # Save the results
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(result)
    print(f"Comparison complete. Results saved to {output_path}")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 4:
    #     print("Usage: python job_comparison.py <file1.json> <file2.json> <output.json>")
    #     sys.exit(1)
    output_path1 = './job_lists_1.json'
    output_path2 = './job_lists_2.json'
    find_new_jobs(output_path1, output_path2)
    
    # find_new_jobs(sys.argv[1], sys.argv[2], sys.argv[3])