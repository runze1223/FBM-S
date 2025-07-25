from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_PEMS,Basis_function,Basis_ETT_hour,Basis_ETT_minute, \
    Dataset_Solar
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import numpy as np 
import os

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
}

data_basis = {
    'ETTh1': Basis_ETT_hour,
    'ETTh2': Basis_ETT_hour,
    'ETTm1': Basis_ETT_minute,
    'ETTm2': Basis_ETT_minute,
    'custom': Basis_function,
}




def data_provider(args, flag):
    Data = data_dict[args.data]

    if args.embed == 'timestamp':
        timeenc = 3
    elif args.embed != 'timeF':
        timeenc = 0
    else:
        timeenc = 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = args.drop_last
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


def basis_provider(args, flag):
    basis_loader = data_basis[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if args.embed != 'timeF':
        timeenc = 1
    elif args.embed != 'timestamp':
        timeenc = 3
    else:
        timeenc = 0

    freq = args.freq

    folder_path = './basis/' + args.data_path + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    basis = basis_loader(
    root_path=args.root_path,
    data_path=args.data_path,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
        )
    basis_data=basis.generate()

    if args.features == 'M':
            if args.data=="ETTh2":
                basis_data[3][:,:-1]=0

    basis_data[0]=np.array(basis_data[0])
    basis_data[1]=np.array(basis_data[1])
    basis_data[2]=np.array(basis_data[2])
    basis_data[3]=np.array(basis_data[3])

    np.save(folder_path + 'basis0.npy', basis_data[0])
    np.save(folder_path + 'basis1.npy', basis_data[1])
    np.save(folder_path + 'basis2.npy', basis_data[2])
    np.save(folder_path + 'basis3.npy', basis_data[3])

    return basis_data
