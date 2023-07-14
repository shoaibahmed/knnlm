!#/bin/bash

URLS=(
    "https://data.statmt.org/news-crawl/en/news.2020.en.shuffled.deduped.gz"
)
FILES=(
    "news.2020.en.shuffled.deduped.gz"
)

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        elif [ ${file: -4} == ".gz" ]; then
            gunzip $file
        fi
    fi
done
cd ..
