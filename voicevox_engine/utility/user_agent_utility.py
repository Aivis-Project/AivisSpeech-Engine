"""ユーザーエージェント文字列を生成する utility"""

import logging
import os
import platform
from typing import Annotated, Literal, cast

import GPUtil
import psutil
from cpuinfo import get_cpu_info
from pydantic import BaseModel, Field

from voicevox_engine import __version__
from voicevox_engine.logging import logger


class AivisSpeechRuntimeEnvironment(BaseModel):
    """
    AivisSpeech Engine の実行環境情報。
    """

    os_name: Annotated[str, Field(description="OS 名")]
    os_version: Annotated[str, Field(description="OS バージョン")]
    distribution: Annotated[
        str | None,
        Field(description="Linux ディストリビューション名 (存在しない場合は null)"),
    ]
    kernel_version: Annotated[
        str | None,
        Field(description="Linux カーネルバージョン (存在しない場合は null)"),
    ]
    is_docker: Annotated[bool, Field(description="Docker コンテナ環境かどうか")]
    architecture: Annotated[str, Field(description="アーキテクチャ名")]
    cpu_name: Annotated[str, Field(description="CPU 名")]
    gpu_names: Annotated[
        list[str],
        Field(description="検出された GPU 名の一覧 (GPU が存在しない場合は空リスト)"),
    ]
    total_memory_gb: Annotated[
        float,
        Field(description="物理メモリ総量 (GB)"),
    ]
    available_memory_gb: Annotated[
        float,
        Field(description="使用可能な物理メモリ量 (GB)"),
    ]
    inference_type: Annotated[
        Literal["CPU", "GPU"],
        Field(description="音声合成に利用する推論方式"),
    ]


# RuntimeEnvironment は inference_type ごとに計測され、プロセス終了までキャッシュされる
__runtime_environment_cache: dict[str, AivisSpeechRuntimeEnvironment] = {}


def collect_runtime_environment(
    inference_type: Literal["CPU", "GPU"],
) -> AivisSpeechRuntimeEnvironment:
    """
    ユーザーエージェント生成に必要な動作環境情報を構造化して取得する。

    一度収集した結果はキャッシュに格納され、プロセスが終了するまで再利用される。

    Parameters
    ----------
    inference_type : Literal["CPU", "GPU"]
        音声合成に利用する推論方式

    Returns
    -------
    runtime_environment : RuntimeEnvironment
        動作環境情報
    """

    cached_environment = __runtime_environment_cache.get(inference_type)
    if cached_environment is not None:
        return cached_environment

    def get_os_details() -> tuple[str, str, str | None, str | None]:
        """OS 名・バージョン・ディストリビューション・カーネルを取得する。"""
        try:
            raw_os_name = platform.system()
        except Exception as ex:
            logger.error("Failed to get OS name:", exc_info=ex)
            return "Unknown", "Unknown", None, None

        if raw_os_name == "Windows":
            try:
                wv = platform.win32_ver()
                os_version = wv[1] if wv[1] else "Unknown"
                return "Windows", os_version, None, None
            except Exception as ex:
                logger.warning("Failed to get Windows version:", exc_info=ex)
                return "Windows", "Unknown", None, None
        if raw_os_name == "Darwin":
            try:
                ver = platform.mac_ver()[0]
                os_version = ver if ver else "Unknown"
                return "macOS", os_version, None, None
            except Exception as ex:
                logger.warning("Failed to get macOS version:", exc_info=ex)
                return "macOS", "Unknown", None, None
        if raw_os_name == "Linux":
            kernel_version = "Unknown"
            try:
                kernel_version = platform.release()
            except Exception as ex:
                logger.warning("Failed to get Linux kernel version:", exc_info=ex)
            distribution = None
            try:
                with open("/etc/os-release", encoding="utf-8") as file:
                    for line in file:
                        if line.startswith("PRETTY_NAME="):
                            distribution = line.split("=")[1].strip().strip('"')
                            break
            except Exception as ex:
                logger.warning("Failed to read /etc/os-release:", exc_info=ex)
            os_version = distribution if distribution is not None else "Unknown"
            return "Linux", os_version, distribution, kernel_version
        return raw_os_name, "Unknown", None, None

    def get_architecture() -> str:
        """アーキテクチャ情報を取得する。エラー時は 'Unknown' を返す。"""
        try:
            return platform.machine()
        except Exception as ex:
            logger.error("Failed to get architecture information:", exc_info=ex)
            return "Unknown"

    def get_cpu_name() -> str:
        """CPU 名を取得する。エラー時は 'Unknown' を返す。"""
        try:
            cpu_info = get_cpu_info()
            return cast(str, cpu_info.get("brand_raw", "Unknown"))
        except Exception as ex:
            logger.error("Failed to get CPU information:", exc_info=ex)
            return "Unknown"

    def get_gpu_names() -> list[str]:
        """GPU 名一覧を取得する。エラー時は ['Unknown'] を返す。"""
        try:
            os_name = platform.system()
            if os_name == "Windows":
                try:
                    import wmi  # type: ignore

                    w = wmi.WMI()
                    gpus = w.Win32_VideoController()
                    names = [gpu.Name for gpu in gpus if hasattr(gpu, "Name")]
                    return names if names else ["Unknown"]
                except Exception as ex:
                    logger.warning(
                        "Failed to get Windows GPU information:", exc_info=ex
                    )
                    return ["Unknown"]
            if os_name == "Linux":
                try:
                    gpus = GPUtil.getGPUs()
                    names = [gpu.name for gpu in gpus if hasattr(gpu, "name")]
                    return names if names else ["NoGPU"]
                except Exception as ex:
                    logger.warning("Failed to get Linux GPU information:", exc_info=ex)
                    return ["Unknown"]
            return ["Unknown"]
        except Exception as ex:
            logger.error("Failed to get GPU information:", exc_info=ex)
            return ["Unknown"]

    def get_memory_info() -> tuple[float, float]:
        """メモリ情報 (総量・使用可能量) を取得する。"""
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024**3), 1)
        available_gb = round(vm.available / (1024**3), 1)
        return total_gb, available_gb

    def is_docker() -> bool:
        """Docker コンテナ内で実行されているかを判定する。"""
        try:
            if os.path.exists("/.dockerenv"):
                return True
            try:
                with open("/proc/1/cgroup", encoding="utf-8") as file:
                    for line in file:
                        if "docker" in line or "kubepods" in line:
                            return True
            except (FileNotFoundError, PermissionError) as ex:
                logger.debug("Docker check - could not read cgroup file:", exc_info=ex)
            return False
        except Exception as ex:
            logger.error("Failed to check Docker environment:", exc_info=ex)
            return False

    try:
        os_name, os_version, distribution, kernel_version = get_os_details()
        architecture = get_architecture()
        cpu_name = get_cpu_name()
        gpu_names = get_gpu_names()
        total_gb, available_gb = get_memory_info()
        is_docker_env = is_docker()

        # Mac では GPU 情報が取得できないため CPU 名を代替として設定する
        if platform.system() == "Darwin" and (
            len(gpu_names) == 0 or all(name == "Unknown" for name in gpu_names)
        ):
            gpu_names = [cpu_name]

        runtime_environment = AivisSpeechRuntimeEnvironment(
            os_name=os_name,
            os_version=os_version,
            distribution=distribution,
            kernel_version=kernel_version,
            is_docker=is_docker_env,
            architecture=architecture,
            cpu_name=cpu_name,
            gpu_names=gpu_names,
            total_memory_gb=total_gb,
            available_memory_gb=available_gb,
            inference_type=inference_type,
        )
        __runtime_environment_cache[inference_type] = runtime_environment
        return runtime_environment
    except Exception as ex:
        logger.error("Failed to collect runtime environment information:", exc_info=ex)
        total_gb, available_gb = get_memory_info()
        fallback_environment = AivisSpeechRuntimeEnvironment(
            os_name="Unknown",
            os_version="Unknown",
            distribution=None,
            kernel_version=None,
            is_docker=False,
            architecture="Unknown",
            cpu_name="Unknown",
            gpu_names=["Unknown"],
            total_memory_gb=total_gb,
            available_memory_gb=available_gb,
            inference_type=inference_type,
        )
        __runtime_environment_cache[inference_type] = fallback_environment
        return fallback_environment


def generate_user_agent(inference_type: Literal["CPU", "GPU"] = "CPU") -> str:
    """
    ユーザーエージェント文字列を生成する。

    エラーが発生した場合でも、最低限の情報を含むユーザーエージェント文字列を返す。

    Parameters
    ----------
    inference_type : Literal["CPU", "GPU"]
        音声合成に利用する推論方式

    Returns
    -------
    user_agent : str
        ユーザーエージェント文字列
    """

    try:
        runtime_environment = collect_runtime_environment(inference_type)

        os_info = runtime_environment.os_name
        if runtime_environment.os_version != "Unknown":
            os_info = f"{os_info}/{runtime_environment.os_version}"
        else:
            os_info = f"{os_info}/Unknown"
        if (
            runtime_environment.os_name == "Linux"
            and runtime_environment.kernel_version is not None
            and runtime_environment.kernel_version != "Unknown"
        ):
            os_info = f"{os_info} (Kernel: {runtime_environment.kernel_version})"

        gpu_info = ", ".join(runtime_environment.gpu_names)
        if not gpu_info:
            gpu_info = "Unknown"

        if (
            runtime_environment.available_memory_gb is not None
            and runtime_environment.total_memory_gb is not None
        ):
            mem_info = (
                f"{runtime_environment.available_memory_gb}GB:"
                f"{runtime_environment.total_memory_gb}GB"
            )
        else:
            mem_info = "Unknown"

        docker_flag = " Docker;" if runtime_environment.is_docker is True else ""

        user_agent = (
            f"AivisSpeech-Engine/{__version__} "
            f"({os_info}; {runtime_environment.architecture};"
            f"{docker_flag} "
            f"CPU/{runtime_environment.cpu_name}; "
            f"GPU/{gpu_info}; "
            f"Memory/{mem_info}; "
            f"Inference/{runtime_environment.inference_type})"
        )
        return user_agent

    except Exception as ex:
        # 最悪の場合のフォールバック
        logger.error("Failed to generate user agent string:", exc_info=ex)
        return f"AivisSpeech-Engine/{__version__}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(generate_user_agent("CPU"))
