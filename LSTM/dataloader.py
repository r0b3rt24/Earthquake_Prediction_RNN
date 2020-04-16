class TrainData(Dataset):
    def __init__(self, df, window_size=1000, sequence_len=150):
        self.rows = df.shape[0] // (window_size*sequence_len)
        self.data, self.labels = [], []
        
        for s in range(self.rows):
            seg = df.iloc[s*window_size*sequence_len: (s+1)*window_size*sequence_len]
            x = seg.acoustic_data.values
            y = seg.time_to_failure.values[-1]
            self.data.append(create_X(x))
            self.labels.append(y)
            
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx].astype(np.float32)),
            self.labels[idx]
        )


def feature_extraction(time_step):
    return np.c_[time_step.mean(axis=1), 
                 np.percentile(np.abs(time_step), q=[0, 25, 50, 75, 100], axis=1).T,
                 time_step.std(axis=1)]



def create_X(x, window_size=1000, seq_len=150):
    X = x.reshape(seq_len, -1)
    return np.c_[feature_extraction(X),
                 feature_extraction(X[:, -window_size // 10:]),]