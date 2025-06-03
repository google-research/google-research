echo "translation error:" `grep error $1 | sort -k2 -n | gawk 'BEGIN{n=0;}{c[n++]=$2}END{if (n % 2) print c[(n-1)/2],n; else print c[n/2-1],n}'`
echo "rotation error:" `grep error $1 | sort -k3 -n | gawk 'BEGIN{n=0;}{c[n++]=$3}END{if (n % 2) print c[(n-1)/2],n; else print c[n/2-1],n}'`
echo "overall time:" `grep "overall_time" $1 | sort -k5 -n | gawk '{c[NR]=$5}END{if (NR % 2) print c[(NR-1)/2],NR; else print c[NR/2-1],NR}'`
