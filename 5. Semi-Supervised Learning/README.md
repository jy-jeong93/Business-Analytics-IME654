## Semi-supervised learning - FixMatch 튜토리얼
이번 튜토리얼에서는 semi-supervised learning 방법론 중 FixMatch를 사용하여 WM811k 웨이퍼 빈 맵 데이터셋에 적용한다.


### WM811k 웨이퍼 빈 맵 데이터셋 소개([출처](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map))

해당 데이터셋은 실제 반도체 웨이퍼 빈 맵 불량 패턴을 하기 위한 데이터셋으로써 총 811,457개의 이미지 데이터셋이다.
또한, 다량의 unlabeled 데이터(78.7%, 638,507개)와 소량의 labeled 데이터(21.3%, 172,950개)로 구성되어있다.

![image](https://user-images.githubusercontent.com/115562646/209644138-da739f69-9615-4ab5-b92c-7b6ef33dc961.png)


단순히 labeled 데이터만 사용하는 supervised learning보다 semi-supervised learning을 통해 분류 성능 개선을 기대할 수 있을 것이며, 동시에 데이터 레이블링 과정에서 발생하는 시간과 비용 문제를 개선할 수 있을 것이다.

![image](https://user-images.githubusercontent.com/115562646/209645858-b2ac4a38-af75-4c97-9a11-b8d11ab8823d.png)


#### (1) Augmentation 정의
Fixmatch 방법론에서는 이미지에 대한 weak augmentation, strong augmentation을 정의해야 한다. 하지만 wm811k데이터셋은 RGB 3차원의 컬러 이미지가 아닌 1차원의 gray scale 이미지이다. 따라서 해당 데이터셋 특성에 맞게 augmentation을 임의로 정의하였다.
아래 그림은 augmentation에 대한 예시이며, strong augmentation에 cutout을 기본적으로 사용한 이유는 FixMatch 본 논문에서 사용했던 strong augmentation 기법 중 유일하게 적용이 가능한 기법이기 때문이다.

![image](https://user-images.githubusercontent.com/115562646/209646589-625bd8df-4603-4f3f-b241-4a50f0b27b28.png)


#### (2) 평가 지표 정의(Macro F1-score)
해당 데이터셋은 공정 데이터셋이므로 정상 패턴이 불량 패턴에 비해 압도적으로 많다. 따라서 단순 accuracy를 사용하는 것이 아닌 Macro F1-score를 평가 지표로써 사용하였다.
단순 semi-supervised learning을 불균형 데이터셋에 적용하면 여러 성능 저하 이슈들이 생기지만, Semi-supervised learning 방법론 적용이 목적인 본 튜토리얼과 어울리지 않으므로 생략하였다.

![image](https://user-images.githubusercontent.com/115562646/209646976-3b5e0687-fc48-4797-b791-304cf26d9203.png)
![image](https://user-images.githubusercontent.com/115562646/209647004-7fe6b567-cfd4-44e3-a399-a91e42ed174f.png)


#### (3) 데이터셋 기본 구축
([출처](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map))에서 제공하는 LSWMD.pkl 파일을 파이썬 실행 경로상에 위치시킨 후, process_wm811k.py를 실행한다.
Labeled, unlabeled 데이터셋이 설정한 대로 train:valid:test = 8:1:1 비율로 나누어져서 각각의 패턴별로 폴더 상에 구축된다.



### wm811k_fixmatch.py

##### Train part
- Labeled 데이터셋과 unlabeled 데이터셋 수가 다르므로 val_iteration을 지정해서 해당 iteration값까지 학습이 진행되고 validation을 하는 방식

```python
def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda, desi_p):
    
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
            
            
        batch_size = inputs_x.size(0)
        
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.type(torch.int64).view(-1, 1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        
        with torch.no_grad():
                        
            outputs_u, _ = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)
            
        max_p, p_hat = torch.max(targets_u, dim=1)

        select_mask1 = max_p.ge(args.tau)
        select_mask2 = torch.rand(batch_size).cuda() < desi_p[p_hat]
        select_mask = select_mask1 * select_mask2
        select_mask = select_mask.float()
        p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
        all_inputs = torch.cat([inputs_x, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
```

##### Validate part
 - Train part에서 지정한 val_iteration이 되면 작동하는 part이며, 평가 지표를 accuracy가 아닌 macro f1-score를 사용

```python
def validate(valloader, model, criterion, use_cuda, mode):
    
    # switch to evaluate mode
    model.eval()
    
    classwise_TP = torch.zeros(num_class)
    classwise_FP = torch.zeros(num_class)
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)
    if use_cuda:
        classwise_TP = torch.zeros(num_class).cuda()
        classwise_FP = torch.zeros(num_class).cuda()
        classwise_correct = classwise_correct.cuda()
        classwise_num = classwise_num.cuda()
        section_acc = section_acc.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):


            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # classwise prediction
            pred_label = outputs.max(1)[1]
            assert pred_label.shape == targets.shape
            confusion_matrix = torch.zeros(num_class, num_class).cuda()
            for r, c in zip(targets, pred_label):
                confusion_matrix[r][c] += 1
                
            classwise_num += confusion_matrix.sum(dim=1)
            classwise_TP += torch.diagonal(confusion_matrix, 0)
            diag_mask = torch.eye(num_class).cuda()
            # 순서 주의 : confusion_matrix가 masked_fiil_(diag_mask.bool(), 0)을 통해서 대각원소 0으로 교체돼서 저장됨
            classwise_FP += confusion_matrix.masked_fill_(diag_mask.bool(), 0).sum(dim=0)
            

    print('classwise_num')
    print(classwise_num)  
    print('classwise_FP')
    print(classwise_FP)
    print('classwise_TP')
    print(classwise_TP)
    classwise_precision = (classwise_TP / (classwise_TP + classwise_FP))
    classwise_recall = (classwise_TP / classwise_num)
    classwise_f1 = torch.nan_to_num((2*classwise_precision*classwise_recall) / (classwise_precision + classwise_recall))
    if use_cuda:
        classwise_precision = classwise_precision.cpu()
        classwise_recall = classwise_recall.cpu()
        classwise_f1 = classwise_f1.cpu()

    return (classwise_precision.numpy(), classwise_recall.numpy(), classwise_f1.numpy())


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar', is_best=False):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_model.pth.tar'))

    return (section_acc.numpy(), classwise_num.numpy(), classwise_precision.numpy(), classwise_recall.numpy(), classwise_f1.numpy())
```


##### Augmentation 조합 별 FixMatch 성능 결과

|  Encoder   
Architecture|    Weak    |   Strong   |
|:-----------------:|:----------:|:----------:|
| Accuracy   |    0.8793  |   0.8278   |
|Architecture|Augmentation|Augmentation|
