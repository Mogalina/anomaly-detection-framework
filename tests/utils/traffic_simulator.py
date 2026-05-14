from locust import HttpUser, task, between

class TrainTicketUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def view_tickets(self):
        self.client.get("/api/v1/tickets")

    @task(1)
    def book_ticket(self):
        self.client.post("/api/v1/book", json={"routeId": "123", "userId": "user1"})