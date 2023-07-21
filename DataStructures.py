import torch
import math
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_1col(Q, conv) -> torch.Tensor:
    """perform the convolution if the output is one column
    Args:
        Q (torch.Tensor): The tensor of to perform the convolution on
        conv (torch.nn.modules.conv.Conv1d): the convolution to perform.
    Raises:
        Value error: if the output of the convolution isn't one column 
    Returns:
        torch.Tensor: the output, needs to be one columns. meaning shape (batch_size, conv.out_channels, 1) or if conv.kernel !=3
    """
    W = conv.weight  # the weights of the convolution
    b = conv.bias  # the bias of the convolution
    dilation = conv.dilation[0]  # the dilation of the convolution

    def conv_func(x):
        return (torch.sum(W*x, (1, 2))+b).reshape(1, -1, 1)

    if Q.size(0) == 1:
        return (torch.sum(W*Q[:, :, ::dilation], (1, 2))+b).reshape(1, -1, 1)
    l1 = [conv_func(q) for q in Q[:, :, ::dilation]]
    return torch.cat(l1, 0).detach()


class ModelRecreate:
    def __init__(self, model, w_max, frame_gen, extractor, normalize, val_augmentation) -> None:
        self.top = RefinementStage(
            model, w_max, model.num_R-1, frame_gen, extractor, normalize, val_augmentation)

    def next(self) -> list:
        """
        returns the predictions of the next frame 
        """
        outs = self.top.next()
        return [o.reshape(1, -1).detach() for o in outs]


class PredictionLayer:
    def __init__(self, model, level, w_max, PG, above_look_ahead=0, batch_size=1) -> None:
        self.level = level  # starts with 0
        self.w_max = w_max  # like in article
        self.batch_size = batch_size

        # define all of the convolutions of the layer
        self.conv_1 = model.PG.conv_dilated_1[level].Dilated_conv
        self.conv_2 = model.PG.conv_dilated_2[level].Dilated_conv
        self.fusion = model.PG.conv_fusion[level]
        self.dropout = model.PG.dropout

        # constants
        self.num_f_maps = self.conv_1.in_channels
        self.dilation_1 = self.conv_1.dilation[0]
        self.dilation_2 = self.conv_2.dilation[0]

        DFW = min(w_max, max(self.dilation_1, self.dilation_2))
        # like in the article
        self.look_ahead = above_look_ahead + DFW
        self.future_t = self.look_ahead

        # maximum length needed to calculate the convolutions
        self.max_length = max(2*self.dilation_1, 2*self.dilation_2)+1

        if level != 0:
            self.prev_layer = PredictionLayer(
                model, level-1, w_max, PG, above_look_ahead + DFW, batch_size=batch_size)
        if level == 0:
            self.PG = PG

        # initialized as [inf], will contain the index of the last frame
        self.T = PG.T

        self.queue = torch.tensor([])  # here, the queue is without padding
        self.create_first_queue()
        if level == model.PG.num_layers-1:
            layer = self
            while level > 0:
                layer = layer.prev_layer
                level -= 1
                layer.queue = layer.queue[:, :, -layer.max_length:]

    def get_init_left_padding(self, dilation: int) -> torch.Tensor:
        """
        gets the left padding in the initilization
        Args:
            dilation (int): the dilation of the convolotion it is used in 
        Returns:
            torch.Tensor: a tensor of zeros in the correct size
        """
        window = self.w_max
        if window > dilation:
            window = dilation
        return torch.zeros(
            self.batch_size, self.num_f_maps, 2*dilation - window, device=device)

    def create_first_queue(self) -> None:
        """
        create the first queue based on the previous layers queue.
        **WITHOUT** the padding
        """
        if self.level == 0:
            frames = []
            for _ in range(self.look_ahead+1):  # get the frames from the PG stage
                frames.append(self.PG.get_from_below())
            self.queue = torch.cat(frames, dim=2)
        else:
            prev_queue = self.prev_layer.queue
            # calc padding for both convolutions
            padding1 = self.get_init_left_padding(self.prev_layer.dilation_1)
            padding2 = self.get_init_left_padding(self.prev_layer.dilation_2)
            # the queues with the padding
            prev_q1 = torch.cat((padding1, prev_queue), 2)
            prev_q2 = torch.cat((padding2, prev_queue), 2)
            # perform convolutions
            q1 = self.prev_layer.conv_1(prev_q1)[
                :, :, :self.look_ahead+1]
            q2 = self.prev_layer.conv_2(prev_q2)[
                :, :, :self.look_ahead+1]
            residual = prev_queue[:, :, :self.look_ahead+1]

            conv_output = F.relu(
                self.prev_layer.fusion(torch.cat((q1, q2), 1)))
            self.queue = conv_output+residual

    def insert_next_frame(self) -> None:
        """
        get the features from below and update the queue
        """
        self.future_t += 1
        if self.future_t <= self.T[0]:  # if not passed the end of the video
            if self.level == 0:
                features = self.PG.get_from_below().detach()

            else:
                features = self.prev_layer.next().detach()
        else:  # if passed the end
            features = torch.zeros(
                self.batch_size, self.num_f_maps, 1, device=device)

        self.queue = torch.cat((self.queue, features), 2)
        self.queue = self.queue[:, :, -self.max_length:]

    def add_left_padding(self, Q: torch.Tensor, length: int) -> torch.Tensor:
        """
        adds the left padding to Q so if Q is smaller then 'size', 
        the size of the output will be length
        Args:
            Q (torch.Tensor): the tensor to add the left padding
            length (int): the size of the output 
        Returns:
            torch.Tensor: (padding of zeros, Q)
        """
        # number of columns we need to add
        left_padding_size = max(0, length-Q.size(2))
        # vector of 0s of this size
        left_padding2 = torch.zeros(
            self.batch_size, self.num_f_maps, left_padding_size, device=device)
        return torch.cat((left_padding2, Q), 2)

    def convolve(self) -> torch.Tensor:
        """
        perform the convolutions, residual, dropout and relu needed to get the next vector
        Returns:
            torch.Tensor: the convolution output (WITH the residual) 
        """
        Q = self.queue
        # decide how to add the padding and what to take from the Q
        if self.dilation_2 >= self.dilation_1:
            Q2 = self.add_left_padding(Q, self.max_length)

            # indentation as defined on the article
            indentation = min(self.w_max, self.dilation_2) - \
                min(self.w_max, self.dilation_1)

            # clip the queue to the right vectors based on the indentation and add padding
            if indentation == 0:
                Q1 = Q2
            else:
                Q1 = Q2[:, :, : -indentation]
        else:
            Q1 = self.add_left_padding(Q, self.max_length)  # add the padding

            # indentation as defined on the article
            indentation = min(self.w_max, self.dilation_1) - \
                min(self.w_max, self.dilation_2)

            # clip the queue to the right vectors based on the indentation and add padding
            if indentation == 0:
                Q2 = Q1
            else:
                Q2 = Q1[:, :, : -indentation]

        Q1 = Q1[:, :, -self.dilation_1*2-1:]  # clip to the needed size
        Q2 = Q2[:, :, -self.dilation_2*2-1:]  # clip to the needed size

        # perform the convolutions
        out1 = conv_1col(Q1, self.conv_1)
        out2 = conv_1col(Q2, self.conv_2)
        if self.batch_size != 1:
            f = self.fusion(torch.cat((out1, out2), 1))
        else:
            f = conv_1col(torch.cat((out1, out2), 1), self.fusion)

        f = F.relu(f)
        f = self.dropout(f)

        # find the residual
        residual = Q[:, :, -min(self.w_max, max(
            self.dilation_1, self.dilation_2))-1]
        residual = residual.reshape(self.batch_size, -1, 1)

        return f + residual

    def next(self) -> torch.Tensor:
        """
        updates the layer and return the next vector
        Returns:
            torch.Tensor: the features of the next time
        """
        self.insert_next_frame()
        return self.convolve()


class PgStage:
    def __init__(self, model, w_max, frame_gen, extractor, normalize, val_augmentation) -> None:
        self.val_augmentation = val_augmentation
        self.normalize = normalize

        self.future_t = 0
        self.t = 0
        # currently at inf. when we will reach the end of the frames, will update
        self.T = [math.inf]

        self.w_max = w_max

        self.extractor = extractor
        self.frame_gen = frame_gen
        try:
            self.batch_size = frame_gen.batch_size
        except AttributeError:
            self.batch_size = 1
        # as defined in 'model.py'
        self.num_f_maps = model.Rs[0].layers[0].conv_1x1.in_channels
        self.conv_1x1_in = model.PG.conv_1x1_in  # convolution in the beginning

        self.conv_outs = model.PG.conv_outs  # convolutions in the end of the
        self.top_layer = PredictionLayer(
            model, model.PG.num_layers-1, w_max, self, batch_size=self.batch_size)

    def get_from_below(self) -> torch.Tensor:
        """
        returns the vector from the stage below and perform convolutions like in the beginning of the stage
        """

        if self.future_t > self.T[0]:  # if passed the end
            return torch.zeros(
                self.batch_size, self.conv_1x1_in.out_channels, 1, device=device)

        current_frame = self.frame_gen.next()

        if current_frame is None:
            self.T[0] = self.future_t - 1
            self.future_t += 1
            return torch.zeros(
                self.batch_size, self.conv_1x1_in.out_channels, 1, device=device)

        self.future_t += 1

        with torch.no_grad():
            current_frame = self.val_augmentation([current_frame])
            frame_tensor = transforms.ToTensor()(current_frame[0]).to(device)
            frame_tensor = self.normalize(frame_tensor)
            frame_tensor = frame_tensor.view(1, *frame_tensor.size())
            features = self.extractor(frame_tensor)[1]

        return self.conv_1x1_in(features.view(1, -1, 1))

    def next(self) -> list:
        """
        returns the features of the next frame after the entire refinement stage
        """
        if self.t > self.T[0]:  # if passed the end
            raise ValueError(f"video ended, nuber of frames is {self.T[0]+1}")

        if self.t == 0:  # if just started
            f = self.top_layer.convolve()
        else:
            f = self.top_layer.next()
        self.t += 1
        outs = []
        # perform convolution for each task
        for conv_out in self.conv_outs:
            outs.append(conv_out(f))
        return outs


class RefinementStage:
    def __init__(self, model, w_max, stage_num, frame_gen, extractor, normalize, val_augmentation) -> None:
        if stage_num != 0:
            self.prev_stage = RefinementStage(
                model, w_max, stage_num-1, frame_gen, extractor, normalize, val_augmentation)
        else:
            self.prev_stage = PgStage(
                model, w_max, frame_gen, extractor, normalize, val_augmentation)
        self.is_top = True if stage_num == len(model.Rs)-1 else False
        try:
            self.batch_size = frame_gen.batch_size
        except AttributeError:
            self.batch_size = 1
        self.t = 0
        self.future_t = 0
        self.T = self.prev_stage.T         # not known from the start - will update
        # it is an array so when updated for one, will update for all

        # convolution in the beginning of the stage
        self.conv_1x1 = model.Rs[stage_num].conv_1x1

        # define the top layer
        self.top_layer = RefinementLayer(
            model, len(model.Rs[0].layers)-1, w_max, stage_num, self, batch_size=self.batch_size)
        self.conv_outs = model.Rs[stage_num].conv_outs

    def next(self) -> list:
        """
        returns the features of the next frame after the entire refinement stage.
        """
        outs = []
        if self.t != 0:
            # get the next features from the top layer
            f = self.top_layer.next()
        else:
            f = self.top_layer.convolve()

        self.t += 1

        # do the convolution for each task
        for conv_out in self.conv_outs:
            outs.append(conv_out(f))

        return outs
        #out = torch.cat(outs, 1)

        # return out

    def get_from_below(self) -> torch.Tensor:
        """
        gets the vector from the stage below and perform convolution and softmax - 
        as in the beginning of each stage
        """
        if self.future_t > self.T[0]:
            x = torch.zeros(self.batch_size,
                            self.conv_1x1.in_channels, 1, device=device)
        else:
            outs = self.prev_stage.next()
            x = torch.cat(outs, 1)

        x = F.softmax(x, dim=1)
        return self.conv_1x1(x)


class RefinementLayer:
    def __init__(self, model, level, w_max, stage_num, R_stage, above_look_ahead=0, batch_size=1) -> None:
        self.t = 0  # the number of the current frame
        self.R_stage = R_stage  # the Refinement stage we are in
        self.batch_size = batch_size
        L_R = len(model.Rs[0].layers)
        self.num_f_maps = model.Rs[0].layers[0].conv_1x1.in_channels

        # the Direct Future Window of the layer
        DFW = min(w_max, 2 ** level)
        self.look_ahead = above_look_ahead + DFW

        if level != 0:
            self.prev_layer = RefinementLayer(
                model, level-1, w_max, stage_num, R_stage, self.look_ahead, batch_size=batch_size)

        self.T = R_stage.T  # not known from the start - will update
        # it is an array so when updated for one, will update for all

        self.level = level
        self.max_length = 1+2**(self.level+1)

        self.w_max = w_max
        self.batch_size = batch_size
        padding_dim = 2**level
        if self.w_max > padding_dim:
            w_max = padding_dim

        # define the convolutions
        self.conv_dilated = model.Rs[stage_num].layers[level].conv_dilated
        self.conv_1x1 = model.Rs[stage_num].layers[level].conv_1x1
        self.dropout = model.Rs[stage_num].layers[level].dropout

        # calculate the padding needed
        self.padding_left = torch.zeros(
            batch_size, self.num_f_maps, 2*padding_dim - w_max, device=device)

        # set up the queue
        self.queue = torch.tensor([])  # just temp
        self.create_first_queue()
        if self.queue.size(2) != self.padding_left.size(2)+self.look_ahead+1:
            raise ValueError("bad Q")

        # the t of the furthest vector in the queue
        self.future_t = self.look_ahead

        # crop the queues
        if level == L_R-1:
            layer = self
            while level > 0:
                layer = layer.prev_layer
                level -= 1
                layer.queue = layer.queue[:, :, -layer.max_length:]

    def create_first_queue(self) -> None:
        """
        create the first queue based on the previous layers queue.
        **WITH** the padding
        """
        if self.level == 0:
            frames = []
            for i in range(self.look_ahead+1):
                probs = self.R_stage.get_from_below()
                frames.append(probs.detach())
            # the features from the previous stage
            received_from_below = torch.cat(frames, 2)
            self.queue = torch.cat((self.padding_left, received_from_below), 2)
        else:
            prev_queue = self.prev_layer.queue

            # use the convolution of the previous layer
            conv_output = F.relu(
                self.prev_layer.conv_dilated(prev_queue))
            conv_output = self.prev_layer.conv_1x1(conv_output)
            conv_output = self.prev_layer.dropout(conv_output)

            below_padding_size = self.prev_layer.padding_left.size(2)
            # residual = the vectors in the prev Q which are not padding
            residual = prev_queue[:, :, below_padding_size:]
            residual = residual[:, :, :self.look_ahead+1]  # the relevant part

            # final
            self.queue = torch.cat(
                (self.padding_left, conv_output + residual), dim=2)

    def convolve(self) -> torch.Tensor:
        """
        does the convolution based on the queue **WITH** the residual
        Returns:
            torch.tensor: the convolution output
        """
        # get the residual
        residual = self.queue[:, :, -min(self.w_max, 2 **
                                         self.level)-1]
        residual = residual.reshape(self.batch_size, -1, 1)

        # do the convolutions
        out = conv_1col(self.queue, self.conv_dilated)  # dilated
        if self.batch_size == 1:
            out = conv_1col(F.relu(out), self.conv_1x1)  # 1x1
        else:
            out = self.conv_1x1(F.relu(out))  # 1x1

        out = self.dropout(out)  # dropout
        return out+residual

    def insert_next_frame(self) -> None:
        """
        updates the queue to 'insert' the features of the next frame and
        'remove' the features of the first frame
        """
        self.future_t += 1
        if self.future_t <= self.T[0]:  # if not passed the end of the video
            if self.level == 0:
                frame_features = self.R_stage.get_from_below().detach()

            else:
                frame_features = self.prev_layer.next().detach()
        else:  # if passed the end
            frame_features = torch.zeros(
                self.batch_size, self.num_f_maps, 1, device=device)

        Q = torch.cat((self.queue, frame_features), dim=2)  # add to the queue
        self.queue = Q[:, :, -self.max_length:]

    def next(self) -> torch.Tensor:
        """
        updates the layer and return the next vector
        Returns:
            torch.Tensor: the features of the next time
        """
        if self.future_t - self.look_ahead >= self.T[0]:
            raise ValueError(f"video ended, nuber of frames is {self.T[0]+1}")
        self.insert_next_frame()
        return self.convolve()
