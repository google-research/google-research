
domain="winered"
fullpathdata="/[PUT FULL PATH HERE]/Datasets/"
fullpathjava="/[PUT FULL PATH HERE]/"
numsamples=(321 321 321 321 320)
splitref=(1 2 3 4 5)

modeltype=("gf")
modeltypeparams=("generative_forest")

trees=(20)
iterations=(500)

for ((i=0; i<=${#numsamples[@]} - 1; i++)); do
    for ((l=0; l<=${#modeltype[@]} - 1; l++)); do
	for ((j=0; j<=${#iterations[@]} - 1; j++)); do
	    for ((k=0; k<=${#trees[@]} - 1; k++)); do
		dname="\"${domain}\""
		echo ""
		echo "-- generating samples tests on $domain with parameters (#Iterations = ${iterations[j]}, #Trees = ${trees[k]})"
		Java -Xmx12000m Wrapper --algorithm_category=1 --dataset="${fullpathdata}${domain}"/Split_"${splitref[i]}"/"${domain}"_train.csv --dataset_test="${fullpathdata}${domain}"/Split_"${splitref[i]}"/"${domain}"_test.csv '--dataset_spec={"name": "$dname", "path": "'${fullpathdata}${domain}'/Split_'${splitref[i]}'/'${domain}'_train.csv", "label": "Dummy", "task": "Dummy"}' --num_samples="${numsamples[i]}" --work_dir="${fullpathdata}${domain}"/working_dir --output_samples="${fullpathdata}${domain}"/Split_"${splitref[i]}"/"${domain}"_"${modeltype[l]}"_I"${iterations[j]}"_T"${trees[k]}"_generated.csv --output_stats="${fullpathdata}${domain}"/results/"${domain}"_generated_observations.stats '--flags={"iterations" : "'${iterations[j]}'", "force_integer_coding" : "true", "force_binary_coding" : "true", "unknown_value_coding" : "NA", "initial_number_of_trees" : "'${trees[k]}'", "splitting_method" : "boosting", "type_of_generative_model" : "'${modeltypeparams[l]}'"}'  --impute_missing=false --density_estimation=false
	    done
	done
    done
done

