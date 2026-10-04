
from mmdet_one_piece.engine import ValLoop_OP
from mmdet_one_piece.evaluation import Evaluator_OP, CocoMetric_OP

# for train_cfg see configs/schedules
val_cfg = dict(
    type=ValLoop_OP,
    data_root = 'data/',  # update dynamically in train.py
    test_pipeline_name=None,  # must be updated in train.py to the actual pipeline name string
    main_val_dataset = None,  # add dynamically in train.py (one of ms-coco, sama-coco, coco-rem)
    train_val_dataloader = None,  # add dynamically in train.py
    extra_val_datasets=['nao', 'coco-o', 'coco-c'],  # update dynamically in train.py
    fp16=False
    )
test_cfg = val_cfg

_coco_anns = [('ms-coco',   'coco/'      + 'annotations/instances_val2017.json'),
              ('sama-coco', 'sama-coco/' + 'annotations/instances_val2017.json'),
              ('coco-rem',  'coco-rem/'  + 'annotations/instances_valrem.json')]
_coco_anns_tv = [('ms-coco',   'coco/'      + 'annotations/instances_train2017_val.json'),
                 ('sama-coco', 'sama-coco/' + 'annotations/instances_train2017_val.json'),
                 ('coco-rem',  'coco-rem/'  + 'annotations/instances_trainrem_val.json')]

val_evaluator= dict(
       type=Evaluator_OP,
       derive_bbox_from_mask=False,
       data_root = 'data/',   # update dynamically in train.py
       ann_files = {
           'ms-coco': _coco_anns, 'sama-coco': _coco_anns, 'coco-rem': _coco_anns,
           'coco-o': {
               'sketch':   [('sketch',   'ood_coco/' + 'sketch/'   + 'annotations/instances_val2017.json')],
               'weather':  [('weather',  'ood_coco/' + 'weather/'  + 'annotations/instances_val2017.json')],
               'cartoon':  [('cartoon',  'ood_coco/' + 'cartoon/'  + 'annotations/instances_val2017.json')],
               'painting': [('painting', 'ood_coco/' + 'painting/' + 'annotations/instances_val2017.json')],
               'tattoo':   [('tattoo',   'ood_coco/' + 'tattoo/'   + 'annotations/instances_val2017.json')],
               'handmake': [('handmake', 'ood_coco/' + 'handmake/' + 'annotations/instances_val2017.json')],
            },
            'nao': [('nao', 'nao/annotations/instances_all.json')],
            'coco-c': _coco_anns,
            'ocs-coco': [_coco_anns[0]],  # only ms-coco anns
            'train': _coco_anns_tv,
       },
       metrics = {  # NOTE: these will be update dynamically depending on if the model predicts masks or not!
           'ms-coco':   ['bbox', 'segm', 'boundary'],
           'sama-coco': ['bbox', 'segm', 'boundary'],
           'coco-rem':  ['bbox', 'segm', 'boundary'],
           'train':     ['bbox', 'segm', 'boundary'],
           'coco-o':    ['bbox'],
           'nao':       ['bbox'],
           'coco-c':    ['bbox', 'segm', 'boundary'],
           'ocs-coco':  ['bbox', 'segm', 'boundary']
       },
       tide = {
           'ms-coco': True, 'sama-coco': True, 'coco-rem': True, 'train': True, 'coco-o': True, 'nao': True, 'coco-c': True, 'ocs-coco': True
       },
       format_only=False,
       outfile_prefix=None,  # update dynamically in train.py
       backend_args=None  # update dynamically in train.py
)
test_evaluator = val_evaluator
