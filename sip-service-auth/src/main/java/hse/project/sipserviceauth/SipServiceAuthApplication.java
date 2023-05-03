package hse.project.sipserviceauth;

import hse.project.sipserviceauth.model.domain.Order;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.LinkedList;
import java.util.Objects;
import java.util.Queue;
import java.util.UUID;

@SpringBootApplication
@EnableJpaRepositories
public class SipServiceAuthApplication implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOrigins(
                        "http://127.0.0.1:5173",
                        "http://localhost:5173",
                        "https://127.0.0.1:5173",
                        "https://localhost:5173"
                )
                .allowedMethods(
                        "GET",
                        "PUT",
                        "POST",
                        "DELETE",
                        "PATCH",
                        "OPTIONS"
                );
    }

    // Queue of orders that have NOT been processed yet
    public static Queue<Order> orders = new LinkedList<>();

    public static void main(String[] args) {
        SpringApplication.run(SipServiceAuthApplication.class, args);

//        try {
//            String url = "jdbc:postgresql://localhost:5432/db";
//            Connection conn = DriverManager.getConnection(url, "postgres", "20010608Kd");
//            Statement stmt = conn.createStatement();
//            ResultSet rs;
//
//            rs = stmt.executeQuery("SELECT * FROM orders WHERE status = false");
//            while (rs.next()) {
//                Order order = Order.builder()
//                        .id((UUID) rs.getObject("id"))
//                        .url(rs.getString("url"))
//                        .url2(rs.getString("url2"))
//                        .model(rs.getString("model"))
//                        .satellite(rs.getString("satellite"))
//                        .createdAt(rs.getDate("created_at"))
//                        .finishedAt(rs.getDate("finished_at"))
//                        .status(rs.getBoolean("status"))
//                        .result(rs.getString("result"))
//                        .result2(rs.getString("result2"))
//                        .diff(rs.getString("diff"))
//                        .build();
//                orders.add(order);
//            }
//            conn.close();
//        } catch (Exception e) {
//            System.err.println("Got an exception! ");
//            System.err.println(e.getMessage());
//        }

        while (true) {
            try {
                orders = new LinkedList<>();
                String url = "jdbc:postgresql://localhost:5432/db";
                Connection conn = DriverManager.getConnection(url, "postgres", "20010608Kd");
                Statement stmt = conn.createStatement();
                ResultSet rs;

                rs = stmt.executeQuery("SELECT * FROM orders WHERE status = false");
                while (rs.next()) {
                    Order order = Order.builder()
                            .id((UUID) rs.getObject("id"))
                            .url(rs.getString("url"))
                            .url2(rs.getString("url2"))
                            .model(rs.getString("model"))
                            .satellite(rs.getString("satellite"))
                            .createdAt(rs.getDate("created_at"))
                            .finishedAt(rs.getDate("finished_at"))
                            .status(rs.getBoolean("status"))
                            .result(rs.getString("result"))
                            .result2(rs.getString("result2"))
                            .diff(rs.getString("diff"))
                            .build();
                    orders.add(order);
                }
                conn.close();
            } catch (Exception e) {
                System.err.println("Got an exception! ");
                System.err.println(e.getMessage());
            }

            if (orders.size() != 0) {
                System.out.println("Start Python 1");

                Order order = orders.peek();
                String order_url = order.getUrl();
                String order_id = order.getId().toString();
                String param_url = "\"" + order_url + "\"";

                try {
                    ProcessBuilder pb = new ProcessBuilder(
                            "C:/diploma/sip-service-main/venv/Scripts/python.exe",
//                            "C:/diploma/sip-service-main/sip-service-auth/src/main/python/old_main.py",
                            "C:/diploma/sip-service-main/sip-service-auth/src/main/python/etl.py",
                            order_id,
                            param_url
                    );
                    pb.redirectErrorStream(true);
                    Process p = pb.start();
                    p.waitFor();
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Finish Python 1");

                System.out.println("Start Python 2");

                try {
                    ProcessBuilder pb = new ProcessBuilder(
                            "C:/diploma/sip-service-main/venv/Scripts/python.exe",
//                            "C:/diploma/sip-service-main/sip-service-auth/src/main/python/old_main.py",
                            "C:/diploma/sip-service-main/sip-service-auth/src/main/python/predict.py",
                            order_id,
                            param_url
                    );
                    pb.redirectErrorStream(true);
                    Process p = pb.start();
                    p.waitFor();
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Finish Python 2");
                orders.remove();
            }
        }
    }
}
