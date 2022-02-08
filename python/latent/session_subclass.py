from sklearn.preprocessing import MinMaxScaler

class Sess():
    def __init__(self,**settings):
        self.sub_type = settings.get('sub_type','AB')
        self.sub = settings.get('sub',1)

        self.train_grp = settings.get('train_grp',2)
        self.train_scale = settings.get('train_scale',5)
        self.train = settings.get('train','fullallmix4')
        self.cv_type = settings.get('cv_type','manual')
        self.feat_type = settings.get('feat_type','feat')
        self.scaler_load = settings.get('scaler_load',True)
        self.epochs = settings.get('epochs',30)

        self.test_grp = settings.get('test_grp',4)
        self.test = settings.get('test','partposrealmixeven14')

        self.emg_scale = settings.get('emg_scale',1)
        self.scaler = settings.get('scaler',MinMaxScaler(feature_range=(0,1)))
    
    def update(self,**settings):
        for k in settings:
            setattr(self, k, settings[k])