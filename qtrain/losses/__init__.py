import torch
class MultiLabelCrossEntropyLoss(torch.nn.Module):
    def __init__(self, wts):
        super().__init__()
        self.criteria = torch.nn.ModuleList(
            [
                torch.nn.CrossEntropyLoss(
                    torch.Tensor([1, wt])
                )
                for wt in wts
            ]
        )

    def forward(self, outputs_list, targets_list):
        return [
            criteria(output, target)
            for criteria, output, target in zip(
                self.criteria, outputs_list, targets_list
            )
        ]