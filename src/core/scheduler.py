"""
定时调度模块 - 使用APScheduler实现定时任务
"""
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None


@dataclass
class ScheduleConfig:
    """调度配置"""
    hour: int = 6
    minute: int = 0
    recurring: bool = True    # 是否重复执行
    enabled: bool = True      # 是否启用


class GameScheduler:
    """
    游戏定时调度器

    使用APScheduler实现定时启动游戏自动化任务

    功能：
    1. 每日定时执行
    2. 一次性定时执行
    3. 间隔执行
    """

    def __init__(self):
        """
        初始化调度器

        Raises:
            ImportError: 如果APScheduler未安装
        """
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "APScheduler is required for scheduling. "
                "Install it with: pip install apscheduler"
            )

        self.scheduler = BackgroundScheduler()
        self.jobs: Dict[str, any] = {}  # job_id -> job

    def schedule_daily(
        self,
        hour: int,
        minute: int,
        job_func: Callable,
        job_id: Optional[str] = None
    ) -> str:
        """
        调度每日执行的任务

        Args:
            hour: 小时 (0-23)
            minute: 分钟 (0-59)
            job_func: 要执行的函数
            job_id: 任务ID（可选）

        Returns:
            任务ID
        """
        trigger = CronTrigger(hour=hour, minute=minute)

        if job_id is None:
            job_id = f"daily_{hour}_{minute}"

        job = self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            replace_existing=True
        )

        self.jobs[job_id] = job
        return job_id

    def schedule_once(
        self,
        run_time: datetime,
        job_func: Callable,
        job_id: Optional[str] = None
    ) -> str:
        """
        调度一次性执行的任务

        Args:
            run_time: 执行时间
            job_func: 要执行的函数
            job_id: 任务ID（可选）

        Returns:
            任务ID
        """
        trigger = DateTrigger(run_date=run_time)

        if job_id is None:
            job_id = f"once_{run_time.strftime('%Y%m%d_%H%M%S')}"

        job = self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            replace_existing=True
        )

        self.jobs[job_id] = job
        return job_id

    def schedule_interval(
        self,
        seconds: int,
        job_func: Callable,
        job_id: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> str:
        """
        调度间隔执行的任务

        Args:
            seconds: 间隔秒数
            job_func: 要执行的函数
            job_id: 任务ID（可选）
            start_date: 开始时间（可选）

        Returns:
            任务ID
        """
        from apscheduler.triggers.interval import IntervalTrigger

        trigger = IntervalTrigger(seconds=seconds, start_date=start_date)

        if job_id is None:
            job_id = f"interval_{seconds}s"

        job = self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            replace_existing=True
        )

        self.jobs[job_id] = job
        return job_id

    def schedule_cron(
        self,
        cron_expr: str,
        job_func: Callable,
        job_id: Optional[str] = None
    ) -> str:
        """
        使用Cron表达式调度任务

        Args:
            cron_expr: Cron表达式 (如 "0 6 * * *" 表示每天6点)
            job_func: 要执行的函数
            job_id: 任务ID（可选）

        Returns:
            任务ID
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        minute, hour, day_of_month, month, day_of_week = parts

        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day_of_month,
            month=month,
            day_of_week=day_of_week
        )

        if job_id is None:
            job_id = f"cron_{cron_expr.replace(' ', '_')}"

        job = self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id=job_id,
            replace_existing=True
        )

        self.jobs[job_id] = job
        return job_id

    def remove_job(self, job_id: str) -> bool:
        """
        移除任务

        Args:
            job_id: 任务ID

        Returns:
            是否移除成功
        """
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            return True
        return False

    def get_job(self, job_id: str):
        """
        获取任务

        Args:
            job_id: 任务ID

        Returns:
            任务对象
        """
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[dict]:
        """
        列出所有任务

        Returns:
            任务信息列表
        """
        result = []
        for job_id, job in self.jobs.items():
            result.append({
                'id': job_id,
                'next_run_time': job.next_run_time,
                'trigger': str(job.trigger)
            })
        return result

    def start(self) -> None:
        """启动调度器"""
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self, wait: bool = True) -> None:
        """
        停止调度器

        Args:
            wait: 是否等待正在执行的任务完成
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)

    def pause(self) -> None:
        """暂停调度器"""
        if self.scheduler.running:
            self.scheduler.pause()

    def resume(self) -> None:
        """恢复调度器"""
        if self.scheduler.running:
            self.scheduler.resume()

    def is_running(self) -> bool:
        """
        检查调度器是否正在运行

        Returns:
            是否正在运行
        """
        return self.scheduler.running if self.scheduler else False


def create_scheduler_from_config(
    config: ScheduleConfig,
    job_func: Callable
) -> Optional[GameScheduler]:
    """
    从配置创建调度器

    Args:
        config: 调度配置
        job_func: 要执行的函数

    Returns:
        GameScheduler实例，如果未启用则返回None
    """
    if not config.enabled:
        return None

    if not APSCHEDULER_AVAILABLE:
        print("Warning: APScheduler not installed, scheduling disabled")
        return None

    scheduler = GameScheduler()

    if config.recurring:
        scheduler.schedule_daily(
            hour=config.hour,
            minute=config.minute,
            job_func=job_func
        )
    else:
        # 一次性执行，明天指定时间
        now = datetime.now()
        run_time = now.replace(
            hour=config.hour,
            minute=config.minute,
            second=0,
            microsecond=0
        )
        if run_time <= now:
            run_time += timedelta(days=1)

        scheduler.schedule_once(run_time=run_time, job_func=job_func)

    return scheduler
