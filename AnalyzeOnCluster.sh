#!/bin/sh

containsElement () {
  local e
  for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
  return 1
}

blacklist="\(123R0126514\)"
files=$(ls logs | grep "[0-9].*\.\(1\|1r\|2\|2r\|master\)\.exp\.log$" | grep -v $blacklist)
files=('logs/123R0126514.1r.exp.log')
files_l=($files)
nstates=$(seq 2 30)
trials=100
iterations=1000
working_dir=$(pwd)
units="amp_and_mel"
printf "%s\n" "$files"
counter=0
size=${#files_l[@]}
# printf $size
for file in $files
do
    file=$(echo $file | sed "s/\(.*\)\.exp\.log/\1/")
    counter=$(($counter + 1))
    printf "\n***********\n"
    printf "Sending \"$file\" to the cluster, file $counter of $size"
    printf "\n***********\n"
    command="python AnalyzeOnCluster.py $file $working_dir $trials $iterations $units $nstates"
    echo $command
    $command
    if [ $counter != $size ]
    then
        echo "Finished $file, moving on to file #$(($counter+1))"
    else
        echo "Finished analyzing $size files."
    fi
done
# echo $files
