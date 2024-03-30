import torch
import csv
import argparse
# Function to load the results and convert them to CSV
def convert_pt_to_csv(pt_file_path, csv_file_path):
    train_results = torch.load(pt_file_path)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        # Create a CSV writer
        writer = csv.writer(csv_file)
        header = train_results.keys()
        writer.writerow(header)
        for key in header:
            values = train_results[key]
            if values.dim() == 0:  # If values is a scalar
                values = [values.item()]  # Convert it to a list
            else:
                values = values.tolist()  # Convert tensor to list
            writer.writerow(values)
    
    return csv_file_path



            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_files", nargs='+', default=[""])
    parser.add_argument("--dataset", default="ISIC19_raw")
    args = parser.parse_args()
    print("analysis_files       : {}".format(args.analysis_files))      
    

    converted_files = []
    for(i, pt_file_path) in enumerate(args.analysis_files):
        path = "models/{}/{}".format(args.dataset, pt_file_path) 
        csv_file_path = path.replace(".pt", ".csv")
        converted_file = convert_pt_to_csv(path, csv_file_path)
        converted_files.append(converted_file)
    
        