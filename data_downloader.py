import torch, os, engine, classify, utils, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Classifier')
    parser.add_argument('--model_name', type=str, default='VGG16', 
                        help='Name of the model to train')
    return parser.parse_args()

dataset_name = "celeba"
device = "cuda"
root_path = "./target_model"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]

    if model_name == "VGG16":
        if mode == "reg": 
            net = classify.VGG16(n_classes)
        elif mode == "vib":
            net = classify.VGG16_vib(n_classes)
    
    elif model_name == "FaceNet":
        net = classify.FaceNet(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        #utils.weights_init_classifier(net.fc_layer)

    elif model_name == "FaceNet_all":
        net = classify.FaceNet(202599)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        #utils.weights_init_classifier(net.fc_layer)
		
    elif model_name == "FaceNet64":
        net = classify.FaceNet64(n_classes)
        BACKBONE_RESUME_ROOT = os.path.join(root_path, "backbone_ir50_ms1m_epoch120.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        # net.fc_layer.apply(net.weight_init)

    elif model_name == "IR50":
        if mode == "reg":
            net = classify.IR50(n_classes)
        elif mode == "vib":
            net = classify.IR50_vib(n_classes)
        BACKBONE_RESUME_ROOT = "ir50.pth"
        print("Loading Backbone Checkpoint ")
        utils.load_my_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        
    elif model_name == "IR152":
        if mode == "reg":
            net = classify.IR152(n_classes)
        else:
            net = classify.IR152_vib(n_classes)

        BACKBONE_RESUME_ROOT = os.path.join(root_path, "Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        print("Loading Backbone Checkpoint ")
        utils.load_state_dict(net.feature, torch.load(BACKBONE_RESUME_ROOT))
        
    else:
        print("Model name Error")
        exit()

    optimizer = torch.optim.SGD(params=net.parameters(),
							    lr=args[model_name]['lr'], 
            					momentum=args[model_name]['momentum'], 
            					weight_decay=args[model_name]['weight_decay'])
	
    epochs = args[model_name]["epochs"]
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).to(device)

    mode = args["dataset"]["mode"]
    n_epochs = args[model_name]['epochs']
    best_ACC = 0
    print("Start Training!")
	
    if mode == "reg":
        best_model, best_acc = engine.train_reg(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
    elif mode == "vib":
        best_model, best_acc = engine.train_vib(args, net, criterion, optimizer, trainloader, testloader, n_epochs)
	
    torch.save({'state_dict':best_model.state_dict()}, os.path.join(model_path, "{}_{:.2f}_allclass.tar").format(model_name, best_acc))

if __name__ == '__main__':
    
    args_cmd = parse_arguments()
    model_name = args_cmd.model_name

    file = "./config/classify.json"
    args = utils.load_json(json_file=file)
    args['dataset']['model_name'] = model_name

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    os.environ["CUDA_VISIBLE_DEVICES"] = args['dataset']['gpus']
    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    train_file = args['dataset']['train_file_path']
    test_file = args['dataset']['test_file_path']
    _, trainloader = utils.init_dataloader(args, train_file, mode="train")
    _, testloader = utils.init_dataloader(args, test_file, mode="test")

    main(args, model_name, trainloader, testloader)
