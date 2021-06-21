import torch.nn as nn
import torch
from at_learner_core.models.wrappers.losses import get_loss
from at_learner_core.models.wrappers.wrapper import Wrapper
from at_learner_core.models.architectures import get_backbone

from .fp_efficientnet import EfficientNet


class FPWrapper(Wrapper):
    def __init__(self, wrapper_config):
        super().__init__()
        self.wrapper_config = wrapper_config
        self.class_names = ['data','eyes','chin','nose','ear_l','ear_r']
        self._init_backbone()
        self._init_classifiers()
        self.__init_single_classifier('liveness',feature_size=self.feature_size*6)
        self._init_loss()
        
    def _init_backbone(self):
        self.backbone, self.feature_size = EfficientNet.from_name('efficientnet-b0'),320
        
    
    def __init_single_classifier(self, clf_name, feature_size = 320, nclasses = 1):
        setattr(self, clf_name, nn.Linear(feature_size, nclasses))

    def _init_classifiers(self):
        for clf_name in self.class_names: 
            self.__init_single_classifier(clf_name)
        
    def _init_loss(self):
        loss_config = None
        if hasattr(self.wrapper_config, 'loss_config'):
            loss_config = self.wrapper_config.loss_config
            self.criterion = get_loss(self.wrapper_config.loss[0], loss_config=loss_config)
            self.criterion_crops = get_loss(self.wrapper_config.loss[1], loss_config=loss_config)
            print('Initialized',self.wrapper_config.loss[1],'for crops')
        else:
            self.criterion = get_loss('BCE', loss_config=loss_config)
            self.criterion_crops = get_loss('BCE', loss_config=loss_config)
        

        self.loss_weights = self.wrapper_config.loss_weights
        
        
        
        
        
    def forward(self,x):
        input_data = [x['data'],x['eyes'],x['chin'],
                      x['nose'],x['ear_l'],x['ear_r']]
        
        output_dict = {}
        loss = None
        features_list,features_cat = self.backbone(input_data)
        outputs=[self.liveness(features_cat)]
        for i,z in enumerate(self.class_names):
            out = getattr(self,z)(features_list[i])
            outputs.append(out)
        
        for idx, clf_name in enumerate(['liveness','data','eyes','chin','nose','ear_l','ear_r']):
            target = x['liveness']
            loss_weight = self.loss_weights[idx]                       
            output = outputs[idx]
            if idx<2:
                criterion = self.criterion
            else:
                criterion = self.criterion_crops

            current_loss = criterion(output,target)
            if idx==0:
                loss = current_loss
            else:
                loss+= loss_weight*current_loss
                
                
            output_dict[clf_name] = target.detach().cpu().numpy()
            final_output = torch.sigmoid(output)
            output_dict[clf_name+'_output'] = final_output.detach().cpu().numpy()
            output_dict[clf_name+'_loss'] = current_loss.detach().cpu().numpy()
        return output_dict,loss
    
    
    
    def to_parallel(self, parallel_class):
        self.backbone = parallel_class(self.backbone)
        for clf_name in ['data','eyes','chin','nose','ear_l','ear_r','liveness']: 
            setattr(self,clf_name,parallel_class(getattr(self,clf_name)))
        return self