for i in $1*".output"
do
    # Remove last two lines
    head -n -2 $i > "temp.output"
    # Add y values
    python addY.py -i "temp.output" -o "temp2.output"
    #Combine all files into one file
    cat "temp2.output" >> "temp3.output"
done

# Select best features for LSTM model
#python selectK.py -i "temp3.output" -o $1".output"
