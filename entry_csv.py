import csv

# Open the CSV file and read each line
with open('ground_truth_train.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    
    # Open a new CSV file to write the Japanese text
    with open('japanese_text_train.csv', 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)
        
        for row in reader:
            # Skip rows that don't have at least two columns (English and Japanese sentences)
            if len(row) < 2:
                continue

            # Get the Japanese text from the row
            japanese_text = row[1]

            # Write the Japanese text to the new CSV file
            writer.writerow([japanese_text])