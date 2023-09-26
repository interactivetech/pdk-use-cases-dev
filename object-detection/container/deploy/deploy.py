import os
import time

import torch
from determined.experimental import Determined
from determined.pytorch import load_trial_from_checkpoint_path
import json
import torchvision
print("TORCHVISION: {}".format(torchvision.__version__))
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from kserve import KServeClient
from common import (
    upload_model,
    get_version,
    DeterminedInfo,
    KServeInfo,
    ModelInfo,
    check_existence,
    create_inference_service,
    wait_for_deployment,
    parse_args,
)

# =====================================================================================

def generate_model_file(template_file: str,output_file: str, num_classes: int):
    """
    Generate a Python model file based on a template, injecting a value into the template.

    Args:
        output_file (str): The name of the output Python file where the generated code will be saved.
        num_classes (int): The number of classes for the model.

    Example:
        generate_model_file('model-file.py', num_classes=10)
    """
    # Read the template file
    with open(template_file, 'r') as template_file:
        template = template_file.read()
    # print(templ)
#     # Replace the placeholders with the provided values
    updated_template = template.replace('{}', str(num_classes))
    print(updated_template)
    # Save the updated template as the output file
    with open(output_file, 'w') as output:
        output.write(updated_template)
# =====================================================================================

def build_frcnn_model_finetune(num_classes,ckpt=None):
    print("Loading pretrained model from {}...".format(ckpt))
    # load an detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # print("Loading checkpoint...")
    # model=load_model_ddp(model,ckpt)
    # print("Done!")
    return model

# =====================================================================================
def load_model_ddp(loaded_model,model_state_dict):
    '''
    '''
    try:
        loaded_model.load_state_dict(model_state_dict)
    except Exception:
        # If the checkpointed model is non-DDP and the current model is DDP, append
        # module prefix to the checkpointed data
        if isinstance(loaded_model, torch.nn.parallel.DistributedDataParallel):
            print("Loading non-DDP checkpoint into a DDP model.")
            torch.nn.modules.utils._add_prefix_in_state_dict_if_not_present(model_state_dict, "module.")
        else:
            # If the checkpointed model is DDP and if we are currently running in
            # single-slot mode, remove the module prefix from checkpointed data
            print("Loading DDP checkpoint into a non-DDP model.")
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_state_dict, "module."
            )
        loaded_model.load_state_dict(model_state_dict)
    return loaded_model
# =====================================================================================
def create_model_file_and_json(path, template_file, output_file):
    '''
    Generates a Python script and a JSON file for Torch-Model-Archiver.

    Torch-Model-Archiver requires two key inputs:
    1. --model-file: Defines the Faster R-CNN model and specifies the number of classes.
    2. --extra-files: Specifies an index-to-name JSON file to map class names to category IDs.

    Args:
    - path (str): The path to the directory containing the model state dictionary file ('state_dict.pth').
    - template_file (str): The template Python script file used for generating the model file.
    - output_file (str): The name of the output JSON file ('index_to_name.json') for mapping class names.

    The function loads the model's state dictionary from 'path', extracts the 'index_to_name' JSON dictionary
    from the checkpoint, and saves it to 'output_file'. It then generates a model file using 'template_file'
    with the specified number of classes and returns.

    Note: Make sure 'path' contains 'state_dict.pth' with the 'index_to_name' JSON.

    Example usage:
    create_model_file_and_json('/path/to/model', 'template.py', 'index_to_name.json')
    '''
    model_state_dict = torch.load(path + '/state_dict.pth',map_location=torch.device('cpu'))
    # Get JSON saved in checkpoint key: index_to_name
    index_to_name_json = model_state_dict['index_to_name']
    n_classes = len(list(index_to_name_json.keys()))
    print("n_classes:", n_classes)
    # Save the dictionary to a JSON file
    with open(output_file, 'w') as output:
        json.dump(index_to_name_json, output, indent=4)

    # Generate the model file using the template
    generate_model_file(template_file, output_file, num_classes=n_classes)
    del model_state_dict
    return
# =====================================================================================
def create_scriptmodule(det_master, det_user, det_pw, model_name, pach_id):
    print(
        f"Loading model version '{model_name}/{pach_id}' from master at '{det_master}...'"
    )

    if os.environ["HOME"] == "/":
        os.environ["HOME"] = "/app"

    os.environ["SERVING_MODE"] = "true"

    start = time.time()
    client = Determined(master=det_master, user=det_user, password=det_pw)
    version = get_version(client, model_name, pach_id)
    checkpoint = version.checkpoint
    checkpoint_dir = checkpoint.download()
    model_state_dict = torch.load(checkpoint_dir + '/state_dict.pth',map_location=torch.device('cpu'))
    index_to_name_json = model_state_dict['index_to_name']
    n_classes = len(list(index_to_name_json.keys()))
    print("--n_classes:", n_classes)
    ckpt = model_state_dict['models_state_dict'][0]
    model = build_frcnn_model_finetune(num_classes=n_classes,ckpt=None)
    print("Loading Checkpoint...")
    model.load_state_dict(ckpt)
    print("Done!")
    # trial = load_trial_from_checkpoint_path(
    #     checkpoint_dir, map_location=torch.device("cpu")
    # )
    
    end = time.time()
    delta = end - start
    print(f"Checkpoint loaded in {delta} seconds.")

    print(f"Creating ScriptModule from Determined checkpoint...")
    # model = trial.model
    model.eval()
    
    # 9.12.23: get index_to_name dictionary saved in checkpoint
    print("Creating model-xview.py and creating index_to_name.json...")
    create_model_file_and_json(checkpoint_dir,'model-xview-template.py','model-xview.py')
    print("Done!")
    
    # Create ScriptModule
    # m = torch.jit.script(model)

    # Save ScriptModule to file
    # torch.jit.save(m, "scriptmodule.pt")
    torch.save(model.state_dict(),'model.pth')
    print(f"ScriptModule created successfully.")


# =====================================================================================


# def create_mar_file(model_name, model_version):
#     print(f"Creating .mar file for model '{model_name}'...")
#     os.system(
#         "torch-model-archiver --model-name %s --version %s --serialized-file ./scriptmodule.pt --handler ./dog_cat_handler.py --force"
#         % (model_name, model_version)
#     )
#     print(f"Created .mar file successfully.")

def create_mar_file(model_name, model_version):
    print(f"Creating .mar file for model '{model_name}'...")
    os.system(
        "torch-model-archiver --model-name %s --version %s --model-file ./model-xview.py --handler ./fasterrcnn_handler.py --serialized-file ./model.pth --extra-files ./index_to_name.json --force"
        % (model_name, model_version)
    )
    print(f"Created .mar file successfully.")

# =====================================================================================


def create_properties_file(model_name, model_version):
    config_properties = """inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"%s":{"%s":{"defaultVersion":true,"marName":"%s.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}""" % (
        model_name,
        model_version,
        model_name,
    )

    conf_prop = open("config.properties", "w")
    n = conf_prop.write(config_properties)
    conf_prop.close()

    model_files = ["config.properties", str(model_name) + ".mar"]

    return model_files


# =====================================================================================

def main():
    args = parse_args()
    det = DeterminedInfo()
    ksrv = KServeInfo()
    model = ModelInfo("/pfs/data/model-info.yaml")

    if args.google_application_credentials:
        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"
        ] = args.google_application_credentials

    print(
        f"Starting pipeline: deploy-name='{args.deployment_name}', model='{model.name}', version='{model.version}'"
    )

    # Pull Determined.AI Checkpoint, load it, and create ScriptModule (TorchScript)
    create_scriptmodule(
        det.master, det.username, det.password, model.name, model.version
    )
    
    # Create .mar file from ScriptModule
    create_mar_file(model.name, model.version)

    # Create config.properties for .mar file, return files to upload to GCS bucket
    model_files = create_properties_file(model.name, model.version)

    # Upload model artifacts to Cloud  bucket in the format for TorchServe
    upload_model(
        model.name, model_files, args.cloud_model_host, args.cloud_model_bucket
    )

    # Instantiate KServe Client using kubeconfig
    if args.k8s_config_file:
        print(f"Using Configured K8s Config File at {args.k8s_config_file}")
        kclient = KServeClient(config_file=args.k8s_config_file)
    else:
        kclient = KServeClient()

    # Check if a previous version of the InferenceService exists (return true/false)
    replace = check_existence(kclient, args.deployment_name, ksrv.namespace)

    resource_requirements = {"requests": {}, "limits": {}}
    if args.resource_requests:
        resource_requirements["requests"] = dict(
            [i.split("=") for i in args.resource_requests]
        )
    if args.resource_limits:
        resource_requirements["limits"] = dict(
            [i.split("=") for i in args.resource_limits]
        )
    # Create or replace inference service
    create_inference_service(
        kclient,
        ksrv.namespace,
        model.name,
        args.deployment_name,
        model.version,
        replace,
        args.cloud_model_host,
        args.cloud_model_bucket,
        args.tolerations,
        resource_requirements,
        args.service_account_name,
        "v1",
    )
    if args.wait and args.cloud_model_host:
        # Wait for InferenceService to be ready for predictions
        wait_for_deployment(
            kclient, ksrv.namespace, args.deployment_name, model.name
        )

    print(
        f"Ending pipeline: deploy-name='{args.deployment_name}', model='{model.name}', version='{model.version}'"
    )


# =====================================================================================


if __name__ == "__main__":
    main()
