import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
# import os
# Use /mnt/disk2T if available, else fall back to ~/.cache
_hf_root = "/mnt/disk2T/hfcache" if os.path.isdir("/mnt/disk2T") else os.path.expanduser("~/.cache/huggingface")
_tmp_root = "/mnt/disk2T/tmp" if os.path.isdir("/mnt/disk2T") else "/tmp"
os.environ["HF_HOME"] = _hf_root
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_root, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_root, "transformers")
os.environ["XDG_CACHE_HOME"] = _hf_root
os.environ["TMPDIR"] = _tmp_root


def _resolve_legacy_checkpoint_path(path):
    if not path:
        return path
    if os.path.exists(path):
        return path
    if "duogesture" in path:
        legacy_path = path  # legacy path resolution
        if os.path.exists(legacy_path):
            logger.warning(f"Checkpoint {path} not found; falling back to legacy path {legacy_path}")
            return legacy_path
    return path


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        if args.ddp:
            # DDP 模式下 rank 对应 GPU id
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            # 单机多卡 / 单卡
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = args.out_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.cache_path+args.out_path+"/"+args.name
        if self.rank==0:
            if self.args.stat == "ts":
                self.writer = SummaryWriter(log_dir=args.out_path + "custom/" + args.name + args.notes + "/")
            else:
                wandb_run_name = args.wandb_name if getattr(args, "wandb_name", None) else args.name[12:] + args.notes
                wandb.init(project=args.project, dir=args.out_path, name=wandb_run_name)
                wandb.config.update(args)
                self.writer = None 
        if args.train_rvq:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        else:
            self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).LMDBNPZDataset(args, "train")
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=args.batch_size,
            shuffle=False if args.ddp else True,   # DistributedSampler is mutually exclusive with shuffle
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if args.ddp else None,
        )
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")
       
        if self.rank == 0:
            if args.train_rvq:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            else:
                self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).PickleDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
            logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{args.model}", fromlist=["something"])

        if args.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            # SyncBatchNorm is NOT used: single-node A100s share NVLink; regular per-GPU
            # BN is correct and avoids the cross-rank allgather in every forward pass.
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=True)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        # mean_1024, std_1024 = _load_hubert_stats(self.args, self.device)
        # _inject_hubert_stats(self.model, mean_1024, std_1024)
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")
            if args.stat == "wandb":
                wandb.watch(self.model)
        
        if args.d_name is not None:
            if args.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
                self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
                                   broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                logger.info(f"init {args.d_name} success")
                if args.stat == "wandb":
                    wandb.watch(self.d_model)
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
           
        if args.e_name is not None:
            """
            bugs on DDP training using eval_model, using additional eval_copy for evaluation 
            """
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            # eval copy is for single card evaluation
            if self.args.ddp:
                self.eval_model = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank) 
            else:
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_copy = getattr(eval_model_module, args.e_name)(args).to(self.rank)
                
            other_tools.load_checkpoints(self.eval_copy, args.data_path+args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path+args.e_path, args.e_name)
            if self.args.ddp:
                self.eval_model = DDP(self.eval_model, device_ids=[self.rank], output_device=self.rank,
                                      broadcast_buffers=False, find_unused_parameters=False)
            self.eval_model.eval()
            self.eval_copy.eval()
            if self.rank == 0:
                logger.info(self.eval_model)
                logger.info(f"init {args.e_name} success")  
                if args.stat == "wandb":
                    wandb.watch(self.eval_model) 
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()
        self.alignmenter = metric.alignment(0.3, 7, self.train_data.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]) if self.rank == 0 else None
        self.align_mask = 60
        self.l1_calculator = metric.L1div() if self.rank == 0 else None
       
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t


    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 165)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 165), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def inverse_selection_tensor_6d(self, filtered_t, selection_array, n):
        new_selected_array = np.zeros((330))
        new_selected_array[::2] = selection_array
        new_selected_array[1::2] = selection_array 
        selection_array = new_selected_array
        selection_array = torch.from_numpy(selection_array).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        if len(filtered_t.shape) == 2:
            original_shape_t = torch.zeros((n, 330)).cuda()
            for i in range(n):
                original_shape_t[i, selected_indices] = filtered_t[i]
        elif len(filtered_t.shape) == 3:
            bs, n, _ = filtered_t.shape
            original_shape_t = torch.zeros((bs, n, 330), device='cuda')
            expanded_indices = selected_indices.unsqueeze(0).unsqueeze(0).expand(bs, n, -1)
            original_shape_t.scatter_(2, expanded_indices, filtered_t)
        return original_shape_t

    def train_recording(self, epoch, its, t_data, t_train, mem_cost, lr_g, lr_d=None):
        pstr = "[%03d][%03d/%03d]  "%(epoch, its, self.train_length)
        for name, states in self.tracker.loss_meters.items():
            metric = states['train']
            if metric.count > 0:
                pstr += "{}: {:.3f}\t".format(name, metric.avg)
                if self.rank == 0:
                    self.writer.add_scalar(f"train/{name}", metric.avg, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({name: metric.avg}, step=epoch*self.train_length+its)
        pstr += "glr: {:.1e}\t".format(lr_g)
        if self.rank == 0:
            self.writer.add_scalar("lr/glr", lr_g, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'glr': lr_g}, step=epoch*self.train_length+its)
        if lr_d is not None:
            pstr += "dlr: {:.1e}\t".format(lr_d)
            if self.rank == 0:
                self.writer.add_scalar("lr/dlr", lr_d, epoch*self.train_length+its) if self.args.stat == "ts" else wandb.log({'dlr': lr_d}, step=epoch*self.train_length+its)
        pstr += "dtime: %04d\t"%(t_data*1000)        
        pstr += "ntime: %04d\t"%(t_train*1000)
        pstr += "mem: {:.2f} ".format(mem_cost*len(self.args.gpus))
        logger.info(pstr)
   
    def test_recording(self, dict_name, value, epoch):
        self.tracker.update_meter(dict_name, "test", value)
        _ = self.tracker.update_values(dict_name, 'test', epoch)

@logger.catch(reraise=True)
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # Redirect stdout/stderr for non-rank-0 processes to per-rank log files.
    # IMPORTANT: also dup2 fds 1 & 2 so that C-level CUDA/NCCL error messages
    # (which bypass Python's sys.stderr) are captured in the log files.
    log_dir = os.environ.get("RANK_LOG_DIR", None)
    if rank != 0 and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        _logpath = os.path.join(log_dir, f"rank_{rank}.log")
        _logfile = open(_logpath, "w", buffering=1)
        sys.stdout = _logfile
        sys.stderr = _logfile
        os.dup2(_logfile.fileno(), 1)   # redirect fd 1 (C-level stdout)
        os.dup2(_logfile.fileno(), 2)   # redirect fd 2 (C-level stderr / CUDA errors)
        # Also add a loguru sink pointing to this file so @logger.catch
        # writes exception tracebacks here rather than silently to the default sink.
        import loguru
        loguru.logger.add(_logfile, level="DEBUG")

    # Diagnostic: show what CUDA devices are visible to this rank
    n_visible = torch.cuda.device_count()
    print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}  device_count={n_visible}", flush=True)

    # MUST come before init_process_group so that .cuda() calls inside __init__
    # map to the correct GPU for this rank.
    # When CUDA_VISIBLE_DEVICES=0,1,2,3 (all visible), rank == physical device index.
    # When torchrun restricts each process to one GPU, device_count==1 and we use 0.
    local_device = rank if n_visible > 1 else 0
    torch.cuda.set_device(local_device)
    print(f"[rank {rank}] set_device({local_device})  device={torch.cuda.current_device()}", flush=True)

    import datetime
    _nccl_timeout = datetime.timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", 600)))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                             timeout=_nccl_timeout)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args)
    print(f"[rank {rank}] CustomTrainer.__init__ completed — GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    start_time = time.time()

    # Resume from checkpoint (all ranks must load so DDP weights stay in sync)
    resume_ckpt = getattr(args, "load_ckpt", None)
    if resume_ckpt and str(resume_ckpt) not in ("", "None", "none"):
        if os.path.exists(resume_ckpt):
            logger.info(f"[rank {rank}] Resuming from checkpoint: {resume_ckpt}")
            other_tools.load_checkpoints(trainer.model, resume_ckpt, args.g_name)
            logger.info(f"[rank {rank}] Checkpoint loaded successfully")
        else:
            logger.warning(f"[rank {rank}] Checkpoint '{resume_ckpt}' not found — starting from scratch")
    else:
        logger.info("Training from scratch ...")

    # Synchronise all ranks after (potentially slow) init and checkpoint loading
    # before entering the training loop that requires collective ops.
    print(f"[rank {rank}] reaching dist.barrier()", flush=True)
    dist.barrier()
    print(f"[rank {rank}] passed dist.barrier()", flush=True)

    if args.inference:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints(trainer.model, _resolve_legacy_checkpoint_path(load_ckpt), "duogesture_model")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
            trainer.inference(args.audio_infer_path, out_name=getattr(args, 'out_name', None))
        return
    if args.test_state:
        if rank == 0:
            if load_ckpt := args.load_ckpt:
                if os.path.exists(load_ckpt):
                    logger.info(f"Loading checkpoint from {load_ckpt}")
                    other_tools.load_checkpoints(trainer.model, _resolve_legacy_checkpoint_path(load_ckpt), "duogesture_model")
                    logger.info("Checkpoint loaded successfully")
                    trainer.model.eval()
                    fid = trainer.test(0)
                    exit(0)
                else:
                    logger.warning(f"Checkpoint {load_ckpt} does not exist. Starting training from scratch.")
                    raise FileNotFoundError(f"Checkpoint {load_ckpt} does not exist.")
    # other_tools.load_checkpoints(trainer.model, '/mnt/disk2T/mm_data/zxy/DuoGesture/weights/best_duogesture_base.bin', 'sem_model')
    # other_tools.load_checkpoints(trainer.model, '/mnt/disk2T/mm_data/zxy/DuoGesture/weights/best_duogesture_sparse.bin', 'sem_model')
    
    # trainer.model.eval()
    # fid = trainer.test(400)
    # exit(0)
    start_epoch = getattr(args, "start_epoch", 0)
    print(f"[rank {rank}] entering training loop  start_epoch={start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs+1):
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if epoch != args.epochs:
            if args.ddp: trainer.train_loader.sampler.set_epoch(epoch)
            trainer.tracker.reset()
            trainer.train(epoch)
        if args.debug:
            other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
            other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), args.g_name)
            fid = trainer.test(epoch)

        # if (epoch) % args.test_period == 0 or epoch == 0:
        if epoch == 0 or (epoch > 100) or (epoch <= 100 and epoch % args.test_period == 0):
        # 执行测试逻辑
        # if epoch >=0:
            if rank == 0:
                # 先评测拿到当前 fid（假设 test 返回 fid 浮点数；若返回 dict，请改成拿 dict['fid']）
                fid = trainer.test(epoch)

                # 判断是否最优（FID 越低越好）
                is_best = (fid is not None) and (fid < getattr(trainer, "best_fid", float("inf")))
                if is_best:
                    trainer.best_fid = fid
                    other_tools.save_checkpoints(
                        os.path.join(trainer.checkpoint_path, f"best_{epoch}.bin"),
                        trainer.model, opt=None, epoch=None, lrs=None
                    )
                # Always save a last checkpoint so training can be resumed
                other_tools.save_checkpoints(
                    os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"),
                    trainer.model, opt=None, epoch=None, lrs=None
                )

        # All ranks must synchronise here: rank 0 may have spent minutes in
        # trainer.test() while ranks 1-3 were idle.  Without this barrier,
        # ranks 1-3 would start trainer.train(epoch+1) and hit a DDP allreduce
        # while rank 0 is still in test() → NCCL SIGABRT.
        if args.ddp:
            dist.barrier()

    # ── wandb: training completed normally ─────────────────────────────────
    if rank == 0 and args.stat == "wandb":
        import wandb as _wandb
        best_fid = getattr(trainer, "best_fid", None)
        fid_str = f"Best FGD : {best_fid:.4f}" if best_fid is not None else "Best FGD : n/a"
        _wandb.alert(
            title="Training Complete \u2705",
            text=(
                f"Run    : {args.name}{args.notes}\n"
                f"Epochs : {start_epoch} \u2192 {args.epochs}\n"
                f"{fid_str}\n"
                f"Node   : {__import__('socket').gethostname()}"
            ),
            level=_wandb.AlertLevel.INFO,
        )
        _wandb.finish()

    # if rank == 0:
    #     for k, v in trainer.tracker.values.items():
    #         if trainer.tracker.loss_meters[k]['val'].count > 0:
    #             other_tools.load_checkpoints(trainer.model, os.path.join(trainer.checkpoint_path, f"{k}.bin"), args.g_name)
    #             logger.info(f"inference on ckpt {k}_val_{v['val']['best']['epoch']}:")
    #             trainer.test(v['val']['best']['epoch'])
    #     other_tools.record_trial(args, trainer.tracker)
    #     if args.stat == "ts":
    #         trainer.writer.close()
    #     else:
    #         wandb.finish()
    
            
if __name__ == "__main__":
    # Use setdefault so the calling shell script can override MASTER_PORT
    # to avoid collisions when multiple jobs run on the same node.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "8680")
    args = config.parse_args()

    def _run(rank, world_size, args):
        """Thin wrapper: catches crashes and fires wandb.alert before re-raising."""
        import socket
        try:
            main_worker(rank, world_size, args)
        except Exception as _exc:
            if rank == 0 and getattr(args, "stat", "") == "wandb":
                import wandb as _wandb
                _wandb.alert(
                    title="Training CRASHED \u274c",
                    text=(
                        f"Run  : {args.name}{getattr(args,'notes','')}\n"
                        f"Error: {type(_exc).__name__}: {_exc}\n"
                        f"Node : {socket.gethostname()}\n"
                        f"Ckpt : {getattr(args, 'load_ckpt', 'none')}"
                    ),
                    level=_wandb.AlertLevel.ERROR,
                )
                _wandb.finish(exit_code=1)
            raise

    if args.ddp:
        # ── torchrun mode: LOCAL_RANK is set as an env var ──
        # torchrun creates one process per rank; we run main_worker directly.
        if "LOCAL_RANK" in os.environ:
            rank       = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ.get("WORLD_SIZE", len(args.gpus)))
            _run(rank, world_size, args)
        else:
            # ── legacy mp.spawn fallback ──
            mp.set_start_method("spawn", force=True)
            mp.spawn(
                _run,
                args=(len(args.gpus), args,),
                nprocs=len(args.gpus),
            )
    else:
        _run(0, 1, args)
