
mkdir data
cd data

wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus cornell_movie_dialogs_corpus

cd ..
python3 data_loader.py --config cornell-movie-dialogs
