# predictions_gan

## Structure of Repository:
    * modules (e.g SocialGan, STI-GAN, Social-BiGat)
    * datasets-related scripts (e.g. Waymo, Shift, etc.)
    * visualization
        * scripts related to vis prediction
    * evaluation scripts (e.g. eval_predictions)

## To download waymo DS
    * [Waymo Dataset](https://waymo.com/open/data/motion/ "Waymo Prediction Page")
    * install AWS CLI https://docs.sbercloud.ru/s3/ug/topics/tools__aws-cli.html
    * download Waymo dataset from AWS S3
        * aws --endpoint-url=https://n-ws-4h1s6.s3pd02.sbercloud.ru s3 cp s3://b-ws-4h1s6-mup/waymo
