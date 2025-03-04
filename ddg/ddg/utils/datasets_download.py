import unittest
from ddg.datasets import *


class TestDatasetsDownload(unittest.TestCase):
    ROOT = '/ailab_mat/dataset/DDG/'

    def test_PACS_download(self):
        # ROOT = '/ailab_mat/dataset/DDG/'
        print('test PACS dataset download')
        PACS(root=self.ROOT,
             domains={'ArtPainting', 'Cartoon', 'Photo', 'Sketch'},
             splits={'train', 'val', 'test'},
             download=False)
        

    def test_OfficeHome(self):
        print('test OfficeHome dataset download')
        OfficeHome(root=self.ROOT,
                   domains={'Art', 'Clipart', 'Product', 'RealWorld'},
                   splits={'train', 'val', 'test'},
                   download=False)

    def test_DomainNet_download(self):
        print('test DomainNet dataset')
        DomainNet(root=self.ROOT,
                  domains={'Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'},
                  splits={'train', 'val', 'test'},
                  download=False)


if __name__ == '__main__':
    unittest.main()
