from flask import Flask, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/run-all', methods=['POST'])
def run_all():
    try:
        # 1. analyze_news.py 실행
        analyze_proc = subprocess.run(
            [sys.executable, 'analyze_news.py'],
            capture_output=True, text=True
        )
        analyze_out = analyze_proc.stdout
        analyze_err = analyze_proc.stderr

        # 2. create_daily_summary.py 실행
        summary_proc = subprocess.run(
            [sys.executable, 'create_daily_summary.py'],
            capture_output=True, text=True
        )
        summary_out = summary_proc.stdout
        summary_err = summary_proc.stderr

        return jsonify({
            'analyze_news': {
                'stdout': analyze_out,
                'stderr': analyze_err,
                'returncode': analyze_proc.returncode
            },
            'create_daily_summary': {
                'stdout': summary_out,
                'stderr': summary_err,
                'returncode': summary_proc.returncode
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 