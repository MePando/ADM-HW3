
# We check if the directory where the .tsv files exists, otherwise we download it
if [ ! -d places/ ]; then

  echo "We download the files:"

  # Download the folder with all the .tsv files
  wget https://datasciencemep27.s3.amazonaws.com/places.zip

  # Unzip the places.zip file
  unzip -q places.zip

  # Remove the places.zip file
  rm places.zip
fi

#concatenate the tsv of each scraped place to have only one file mantaining the header of the first file only
head -n 1 places/place_1.tsv > merged.tsv
for F in places/place*.tsv ; do tail -n +2 ${F} >> merged.tsv ; done

#extract the columns of interest for the analysis 
cut -f3,4,8 merged.tsv > columns.tsv
#count the number of rows containing the string corresponding to the name of the country to obtain the number of places
echo 'The number of places in England are:'
grep -c 'England' columns.tsv
echo 'The number of places in Italy are:'
grep -c 'Italy' columns.tsv
echo 'The number of places in Spain are:'
grep -c 'Spain' columns.tsv
echo 'The number of places in France are:'
grep -c 'France' columns.tsv
echo 'The number of places in the United States are:'
grep -c 'United States' columns.tsv

#extract the rows corresponding to places in the selected countries

awk -F"\t" '/England/ {print $0}' columns.tsv > england.tsv
awk -F"\t" '/Italy/ {print $0}' columns.tsv > italy.tsv 
awk -F"\t" '/Spain/ {print $0}' columns.tsv > spain.tsv
awk -F"\t" '/France/ {print $0}' columns.tsv > france.tsv
awk -F"\t" '/United States/ {print $0}' columns.tsv > united_states.tsv

#compute the average of the number of people who visited places in each country

echo 'The average number of people who visited places in England is:'
awk '{s+=$1} END {printf("%0.f\n", s/NR);}' england.tsv
echo 'The average number of people who visited places in Italy is:'
awk '{s+=$1} END {printf("%0.f\n", s/NR);}' italy.tsv 
echo 'The average number of people who visited places in Spain is:'
awk '{s+=$1} END {printf("%0.f\n", s/NR);}' spain.tsv
echo 'The average number of people who visited places in France is:'
awk '{s+=$1} END {printf("%.0f\n", s/NR);}' france.tsv
echo 'The average number of people who visited places in the United States is:'
awk '{s+=$1} END {printf("%.0f\n", s/NR);}' united_states.tsv

#sum all the values corresponding to the number of people wanting to visit places in each country
 
echo 'The total number of people wanting to visit places in England is:'
awk '{s+=$2} END {print s}' england.tsv
echo 'The total number of people wanting to visit places in Italy is:'
awk '{s+=$2} END {print s}' italy.tsv 
echo 'The total number of people wanting to visit places in Spain is:'
awk '{s+=$2} END {print s}' spain.tsv
echo 'The total number of people wanting to visit places in France is:'
awk '{s+=$2} END {print s}' france.tsv
echo 'The total number of people wanting to visit places in the United States is:'
awk '{s+=$2} END {print s}' united_states.tsv
