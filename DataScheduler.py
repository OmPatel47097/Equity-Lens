import schedule
import time
from datetime import datetime
from utils.StockDataManager import StockDataManager
from utils.LoggerManager import LoggerManager


class DataScheduler:
    def __init__(self, update_interval='daily'):
        self.stock_data_manager = StockDataManager()
        self.update_interval = update_interval
        self.logger = LoggerManager.get_logger(__name__)

    def update_data(self):
        try:
            self.logger.info("Starting data update")
            # self.stock_data_manager.download_adj_close()
            self.stock_data_manager.download_histories()
            self.logger.info("Data update completed")
        except Exception as e:
            self.logger.error(f"Error during data update: {e}")

    def initial_fetch(self):
        self.logger.info("Performing initial data fetch")
        self.update_data()

    def schedule_updates(self):
        self.initial_fetch()  # Fetch all data initially

        if self.update_interval == 'daily':
            schedule.every().day.at("00:00").do(self.update_data)
        elif self.update_interval == 'hourly':
            schedule.every().hour.do(self.update_data)
        else:
            self.logger.error(f"Unsupported update interval: {self.update_interval}")

        self.logger.info(f"Scheduled updates with interval: {self.update_interval}")
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    data_scheduler = DataScheduler(update_interval='daily')
    data_scheduler.schedule_updates()
