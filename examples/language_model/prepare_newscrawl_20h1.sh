!#/bin/bash

echo "Downloading file..."
wget https://data.statmt.org/news-crawl/en/news.2020.en.shuffled.deduped.gz

echo "Unziping file..."
gunzip news.2020.en.shuffled.deduped.gz

