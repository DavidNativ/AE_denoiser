# step  : download local copy of the MNIST dataset
# step  : create the dataset, upload files and finalize it
# step  : init task
# step  : train the model
import os

from clearml import PipelineDecorator

@PipelineDecorator.component(return_values=['dataset_path'], cache=True)
def step_one_DL_MNIST(data_url):
    from clearml import StorageManager
    print('==> STEP1: Retrieving MNIST data')
    print("- Creating a local copy")
    dataset_path = StorageManager.get_local_copy(remote_url=data_url)
    print(f'- Local path: {dataset_path}')
    return dataset_path

@PipelineDecorator.component(return_values=['dataset'], cache=True)
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

@PipelineDecorator.component(
    return_values=['train_loader', 'test_loader'], cache=False ) #, monitor_artifacts=['img',])
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
        img = pil.fromarray((img*255).astype(np.uint8))

        #### no previews (artefacts)
        #PipelineDecorator.upload_artifact(name=f'Clean Image No.{i}', artifact_object=img)

        #### previews (debug samples)
        #logger = PipelineController.get_logger()
        #logger.report_image(title="original MNIST", series=f"Clean Image No.{i}", image=img)

        max += 1
        if max > 8:
            break
    return train_loader, test_loader


@PipelineDecorator.pipeline(
    name='MNIST Deco Pipeline', project='tuto', version='0.0.1',
    default_queue='default', pipeline_execution_queue='default'
)
def pipeline():
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

    # # Clear ML integration
    # task = Task.init(project_name=project_name, task_name=task_name)
    # task.connect(hyper_param_dict)

    std = 1.
    mean = 0.

    #steps parameters
    url = "http://yann.lecun.com/exdb/mnist/"
    dataset_name = "Dataset_workshop"
    dataset_project = "03_dataset_MNIST_deco_pipeline"
    project_name = "02_AE_MNIST_denoiser"
    task_name = 'ClearML Task Name',

    #building the pipeline
    dataset_path = step_one_DL_MNIST(url)
    dataset = step_two_create_dataset(dataset_path, dataset_name=dataset_name, dataset_project=dataset_project)
    train_loader, test_loader = step_three_create_dataloaders(dataset, batch_size, dataset_path)


if __name__ == "__main__":
    #PipelineDecorator.debug_pipeline()
    PipelineDecorator.run_locally()
    pipeline()
    print('pipeline completed')
