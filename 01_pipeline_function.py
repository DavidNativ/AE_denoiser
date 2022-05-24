# step  : download local copy of the MNIST dataset
# step  : create the dataset, upload files and finalize it
# step  : init task
# step  : train the model
import os

from clearml import PipelineController

def step_one_DL_MNIST(data_url):
    from clearml import StorageManager
    print('==> STEP1: Retrieving MNIST data')
    print("- Creating a local copy")
    dataset_path = StorageManager.get_local_copy(remote_url=data_url)
    print(f'- Local path: {dataset_path}')
    return dataset_path

def step_two_create_dataset(dataset_path, dataset_name, dataset_project):
    from clearml import Dataset
    print('==> STEP2: Creating dataset')
    print(f"- Creating the ClearML entry {dataset_name}/{dataset_project}")
    dataset = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project)

    print("- Adding the files")
    dataset.add_files(path=dataset_path)
    print("- Uploading")
    # Dataset is uploaded to the ClearML Server by default
    dataset.upload(verbose=True)
    print("- Finalizing")
    dataset.finalize(verbose=True)
    return dataset

def step_three_create_dataloaders(dataset, batch_size, dataset_path):
    from clearml import PipelineController
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import os
    #import matplotlib.pyplot as plt
    import PIL.Image as pil
    import numpy as np

    print(os.getcwd())

    # transformer for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])

    print('- Creating the dataloaders')
    #### datasets & dataloaders
    # preparing the datasets
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data', train=False, download=True, transform=transform),
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(batch_size), shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset[0], batch_size=int(batch_size), shuffle=True, drop_last=True)

    print('- Creating the artefacts')
    max = 0
    for (i, sample) in enumerate(test_loader):
        r = np.random.randint(batch_size)
        img = sample[0][r].view(28, 28).numpy()
        img = pil.fromarray(img)

        logger = PipelineController.get_logger()
        logger.report_image(title="original MNIST", series=f"Clean Image No.{i}", image=img)
        #PipelineController.upload_artifact('img', img)

        max += 1
        if max > 8:
            break
    return train_loader, test_loader

if __name__ == "__main__":
    import argparse

    CLEARML = True

    ##### args
    # Setting Hyperparameters through a dict ....
    hyper_param_dict = {
        #"batch_size": 128,
        "learning_rate": 0.01,
        "checkpoint": 3,
        "epochs":4
    }

    # setting another HP through arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=128)
    args = parser.parse_args()
    batch_size = str(args.batch_size)

    # learning_rate = hyper_param_dict["learning_rate"]
    # checkpoint = hyper_param_dict["checkpoint"]

    # # Clear ML integration
    # task = Task.init(project_name=project_name, task_name=task_name)
    # task.connect(hyper_param_dict)

    std = 1.
    mean = 0.

    # creating the controller
    pipe = PipelineController(
        name='MNIST Pipeline',
        project='tuto',
        version='0.0.1',
        add_pipeline_tags=False,
        auto_version_bump=True
    )

    pipe.add_parameter(
        name='url',
        description='url to pickle file',
        default="http://yann.lecun.com/exdb/mnist/"
    )
    pipe.add_parameter(
        name='dataset_name',
        description='ClearML WebApp Dataset Name',
        default="Dataset_workshop"
    )
    pipe.add_parameter(
        name='dataset_project',
        description='ClearML WebApp Project Name',
        default="02_dataset_MNIST_fun_pipeline"
    )
    pipe.add_parameter(
        name='project_name',
        description='ClearML Project Folder',
        default="02_AE_MNIST_denoiser"
    )
    pipe.add_parameter(
        name='task_name',
        description='ClearML Task Name',
        default="Training inside a pipeline"
    )
    pipe.add_parameter(
        name='batch_size',
        description='Size of the batches',
        default=batch_size
    )
    pipe.add_parameter(
        name='hyper_param_dict',
        description='Dictionary of the hyper parameters',
        default=hyper_param_dict
    )

    #building the pipeline
    pipe.add_function_step(
        name='step_one',
        function=step_one_DL_MNIST,
        function_kwargs=dict(data_url='${pipeline.url}'),
        function_return=['dataset_path'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='step_two',
        function=step_two_create_dataset,
        function_kwargs=dict(
            dataset_path='${step_one.dataset_path}',
            dataset_name='${pipeline.dataset_name}',
            dataset_project='${pipeline.dataset_project}',
        ),
        function_return=['dataset'],
        cache_executed_step=True,
    )

    pipe.add_function_step(
        name='step_three',
        function=step_three_create_dataloaders,
        function_kwargs=dict(
            dataset='${step_two.dataset}',
            dataset_path='${step_one.dataset_path}',
            batch_size='${pipeline.batch_size}',
        ),
        function_return=['train_loader','test_loader'],
        cache_executed_step=False,
        #monitor_artifacts=['img']
    )

    # project_name = '${pipeline.project_name}',
    # task_name = '${pipeline.task_name}',

    #enqueue or locally start
    pipe.set_default_execution_queue('default')
    pipe.start(queue='default')
    #pipe.start_locally(run_pipeline_steps_locally=True)
