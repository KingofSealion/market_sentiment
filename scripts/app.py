from flask import Flask, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/run-all', methods=['POST'])
def run_all():
    """과거 데이터 대용량 처리용: analyze_news.py + create_daily_summary.py 순차 실행"""
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
            'message': 'Batch processing completed',
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

@app.route('/analyze-news', methods=['POST'])
def analyze_news_only():
    """실시간 운영용: 뉴스 분석만 실행 (30분마다)"""
    try:
        proc = subprocess.run(
            [sys.executable, 'analyze_news.py'],
            capture_output=True, text=True
        )
        
        return jsonify({
            'message': 'News analysis completed',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/daily-summary', methods=['POST'])
def daily_summary_only():
    """실시간 운영용: 일일 요약 생성만 실행 (매일 새벽)"""
    try:
        proc = subprocess.run(
            [sys.executable, 'create_daily_summary.py'],
            capture_output=True, text=True
        )
        
        return jsonify({
            'message': 'Daily summary completed',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크용 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'available_endpoints': [
            'POST /run-all - 과거 데이터 일괄 처리',
            'POST /analyze-news - 뉴스 분석만 (실시간용)',
            'POST /daily-summary - 일일 요약만 (실시간용)',
            'GET /health - 상태 확인'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 