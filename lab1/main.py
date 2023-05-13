from recognition.KNNMetricsAnalyzer import check_metrics
from recognition.dataset.Dataset import *

def main():
    check_metrics(1000, False, False, 1, 0.2, 0.4, 0.3, 0.2, blackWhiteNumbers, blackWhiteNumbers)

if __name__ == '__main__':
    main()