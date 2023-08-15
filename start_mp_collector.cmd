conda activate tensortrade
start /B python mp_collect_server.py --gpu 0 --num_worker 10 --num_env 3 --max_steps 50 --port 29550 > .\logs\29550.log
start /B python mp_collect_server.py --gpu 0 --num_worker 10 --num_env 3 --max_steps 50 --port 29551 > .\logs\29551.log
start /B python mp_collect_server.py --gpu 0 --num_worker 10 --num_env 3 --max_steps 50 --port 29552 > .\logs\29552.log
start /B python mp_collect_server.py --gpu 1 --num_worker 10 --num_env 3 --max_steps 50 --port 29553 > .\logs\29553.log
start /B python mp_collect_server.py --gpu 1 --num_worker 10 --num_env 3 --max_steps 50 --port 29554 > .\logs\29554.log
start /B python mp_collect_server.py --gpu 1 --num_worker 10 --num_env 3 --max_steps 50 --port 29555 > .\logs\29555.log