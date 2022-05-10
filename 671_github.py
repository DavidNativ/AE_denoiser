#  Datasets cache folder not shared between users #671

from clearml import *
import os,time

dataset_name = "MNIST"
dataset_project = "AE_denoise"


def create_dataset(dataset_name, dataset_project):
    print("-> Creating the dataset")
    my_dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    )
    print(my_dataset.id)

    #add files
    print("-> Adding files")
    cur_dir = os.getcwd()
    data_dir = os.path.join(cur_dir, "data")
    data_dir = os.path.join(data_dir, "MNIST")
    data_dir = os.path.join(data_dir, "raw")

    my_dataset.add_files(
        path=data_dir,
        wildcard='*.*',
        recursive=True
    )

    #uploading the files
    #the user uses a AWS server but the issue comes from the local cache
    #therefore we can UL on the Cml server
    print('-> Uploading')
    my_dataset.upload(verbose=True)

    #finalizing
    print("-> Finalizing")
    my_dataset.finalize(verbose=True)

    return my_dataset.id


def get_cache(dataset_name, dataset_project):
    print("-> Creating a local cache copy")

    return Dataset.get(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    ).get_local_copy()

if __name__ == '__main__' :
    task = Task.init(project_name='01_AE_MNIST_torch', task_name='04 HPO')

    #create dataset
    #dataset_id = create_dataset(dataset_name, dataset_project)

    #consuming
    dataset_path = get_cache(dataset_name, dataset_project)
    print(dataset_path)

    print("Accessing")
    for i in range(100):
        print(f"{i} {Dataset.get(dataset_name=dataset_name,dataset_project=dataset_project)}")
        #time.sleep(1)



    #deleting
    #Dataset.delete(dataset_id=dataset_id)
