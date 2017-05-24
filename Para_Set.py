import yaml

Para={}
Para['version'] = 'V1'

Para['dataset'] = 'MSLR-WEB10K'
Para['Nfeature'] = 136
Para['Learningrate'] = 0.0001
Para['Nepisode'] = 100
Para['Lenepisode'] = 10

Para_file = open('Para.yml', 'w+')
yaml.dump(Para, Para_file)

Para = yaml.load(file('Para.yml'))
print Para